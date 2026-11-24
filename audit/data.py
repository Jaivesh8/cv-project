import hashlib
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
PROTOCOL_PATH = ROOT / "protocol.json"
IMAGE_BASE = "https://s3.amazonaws.com/images.cocodataset.org/val2014"
IMAGE_NAME = re.compile(r"COCO_val2014_\d{12}\.jpg\Z")
OBJECT_QUESTION = re.compile(r"Is there (?:an? )?(.+?) in the image\?\Z")
PLURAL_TARGETS = {"scissors", "skis"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url, path):
    if path.is_file() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            with urlopen(url, timeout=25) as response:
                payload = response.read()
            if not payload:
                raise ValueError("Empty download")
            tmp.write_bytes(payload)
            tmp.replace(path)
            return path
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def annotations(source, split, config):
    revision = config["annotation_revisions"][source]
    if source == "pope":
        repo, suffix = "RUCAIBox/POPE", f"output/coco/coco_pope_{split}.json"
    else:
        repo, suffix = "YanNeu/RePOPE", f"annotations/coco_repope_{split}.json"
    path = fetch(
        f"https://raw.githubusercontent.com/{repo}/{revision}/{suffix}",
        CACHE / "annotations" / revision / Path(suffix).name,
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return rows, digest(path)


def make_probes(config, pilot):
    rng = random.Random(config["seed"])
    quotas = config["images_per_negative_strategy"]
    pilot_quotas = config["pilot_images_per_negative_strategy"]
    candidates = {}
    hashes = {}
    positives = {}
    for split in quotas:
        original, hashes[f"pope_{split}"] = annotations("pope", split, config)
        corrected, hashes[f"repope_{split}"] = annotations("repope", split, config)
        original_by_id = {row["question_id"]: row for row in original}
        candidates[split] = {}
        for row in sorted(corrected, key=lambda r: r["question_id"]):
            oid, image, question, label = (row[k] for k in ("question_id", "image", "text", "label"))
            old = original_by_id[oid]
            if (old["image"], old["text"]) != (image, question):
                raise ValueError(f"Source mismatch: {split} question {oid}")
            if label not in {"yes", "no"} or not IMAGE_NAME.fullmatch(image):
                raise ValueError(f"Unexpected annotation in {split} question {oid}")
            match = OBJECT_QUESTION.fullmatch(question)
            if not match:
                raise ValueError(f"Unexpected object question: {question!r}")
            probe = dict(probe_id=f"{split}:{oid}", image=image, target=match[1],
                         label=label, original_label=old["label"], split=split)
            if label == "yes" and split == "random":
                positives.setdefault(image, []).append(probe)
            elif label == "no":
                candidates[split].setdefault(image, []).append(probe)

    used = set()
    groups = {False: [], True: []}
    # Each image contributes one present and one absent query. Reserve distinct pilot images first.
    for is_pilot in (True, False):
        counts = pilot_quotas if is_pilot else quotas
        ordered_splits = sorted(counts, key=lambda s: len(candidates[s].keys() & positives.keys()))
        for split in ordered_splits:
            pool = sorted((candidates[split].keys() & positives.keys()) - used)
            if len(pool) < counts[split]:
                raise ValueError(f"Not enough unused images in {split}: {len(pool)}")
            for image in rng.sample(pool, counts[split]):
                negative = rng.choice(candidates[split][image])
                positive = rng.choice(positives[image])
                if positive["target"] == negative["target"]:
                    raise ValueError(f"Conflicting corrected labels on {image}")
                groups[is_pilot].extend((positive, negative))
                used.add(image)
    probes = groups[pilot]
    rng.shuffle(probes)
    payload = json.dumps(probes, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    return probes, hashes, hashlib.sha256(payload).hexdigest()


def download_images(probes):
    names = sorted({probe["image"] for probe in probes})
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = pool.map(lambda name: fetch(f"{IMAGE_BASE}/{name}", CACHE / "images" / name), names)
        return dict(zip(names, paths, strict=True))


def noun_phrase(target):
    if target == "broccoli" or target in PLURAL_TARGETS:
        return f"any {target}"
    article = "an" if target[0] in "aeiou" else "a"
    return f"{article} {target}"


def format_prompt(config, template, target):
    question = config["templates"][template].format(target=noun_phrase(target))
    if template == "W0" and target in PLURAL_TARGETS:
        question = question.replace("Is there ", "Are there ", 1)
    return question + " " + config["answer_instruction"]
