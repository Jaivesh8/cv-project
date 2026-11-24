import hashlib
import json
import os
import platform
import re
import time
from functools import lru_cache

from .data import CACHE, digest, download_images, format_prompt


def parse_answer(raw):
    match = re.match(r"\s*(yes|no)\b", raw, flags=re.IGNORECASE)
    return match[1].lower() if match else "invalid"


def load_predictions(path):
    if not path.exists():
        return {}
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["model"], row["probe_id"], row["arm"])
            if key in rows or parse_answer(row["raw"]) != row["answer"]:
                raise ValueError(f"Duplicate or invalid prediction record: {key}")
            rows[key] = row
    return rows


def run(config, probes, metadata, out, device_name, batch_size, offline):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import transformers
    from PIL import Image, ImageOps
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this PyTorch installation")
    dtype = torch.float16 if device == "cuda" else torch.float32
    torch.manual_seed(config["seed"])
    if device == "cpu":
        torch.set_num_threads(min(6, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    details = {
        **metadata,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device,
        "hardware": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
        "system": platform.platform(),
        "dtype": str(dtype),
        "batch_size": batch_size,
    }
    info = out / "run.json"
    if info.exists() and json.loads(info.read_text(encoding="utf-8")) != details:
        raise ValueError("Run settings differ from existing outputs. Use a new --out directory.")
    info.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(details), flush=True)
    log = out / "predictions.jsonl"
    existing = load_predictions(log)
    work = [(probe, arm) for arm in (*config["templates"], "blank_W0") for probe in probes]
    expected = [(model, p["probe_id"], arm) for model in config["models"] for p, arm in work]
    if all(key in existing for key in expected):
        print("All model predictions already exist.", flush=True)
        return
    if offline:
        paths = {p["image"]: CACHE / "images" / p["image"] for p in probes}
    else:
        paths = download_images(probes)
    image_hashes = {name: digest(path) for name, path in sorted(paths.items())}
    image_digest = hashlib.sha256(json.dumps(image_hashes, sort_keys=True).encode()).hexdigest()
    identity = out / "image_digest.txt"
    if identity.exists() and identity.read_text().strip() != image_digest:
        raise ValueError("Input images differ from the recorded run")
    identity.write_text(image_digest + "\n", encoding="utf-8")
    print(f"image_sha256={image_digest}", flush=True)

    @lru_cache(maxsize=256)
    def image(name):
        with Image.open(paths[name]) as source:
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            rgb.thumbnail((config["image_max_edge"],) * 2, Image.Resampling.LANCZOS)
        return rgb

    def predict(batch, model, processor):
        prompts, images = [], []
        for probe, arm in batch:
            wording = "W0" if arm == "blank_W0" else arm
            question = format_prompt(config, wording, probe["target"])
            message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
            prompts.append(processor.apply_chat_template(message, add_generation_prompt=True))
            rgb = image(probe["image"])
            images.append([Image.new("RGB", rgb.size, (128, 128, 128)) if arm == "blank_W0" else rgb])
        inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
        inputs = {key: (value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device))
                  for key, value in inputs.items()}
        with torch.inference_mode():
            tokens = model.generate(**inputs, do_sample=False, num_beams=1,
                                    max_new_tokens=config["max_new_tokens"], use_cache=True)
        continuations = tokens[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(continuations, skip_special_tokens=True)

    with log.open("a", encoding="utf-8", newline="\n") as stream:
        for name, (model_id, revision) in config["models"].items():
            pending = [(p, arm) for p, arm in work if (name, p["probe_id"], arm) not in existing]
            if not pending:
                continue
            processor = AutoProcessor.from_pretrained(model_id, revision=revision, use_fast=False,
                                                      local_files_only=offline, token=False)
            processor.tokenizer.padding_side = "left"
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, revision=revision, torch_dtype=dtype, attn_implementation="sdpa",
                local_files_only=offline, token=False,
            ).to(device).eval()
            print(f"{name}: {len(pending)} pending responses", flush=True)
            started = time.perf_counter()
            for offset in range(0, len(pending), batch_size):
                batch = pending[offset:offset + batch_size]
                before = time.perf_counter()
                answers = predict(batch, model, processor)
                elapsed = (time.perf_counter() - before) / len(batch)
                for (probe, arm), raw in zip(batch, answers, strict=True):
                    row = dict(model=name, probe_id=probe["probe_id"], arm=arm, raw=raw,
                               answer=parse_answer(raw), seconds=round(elapsed, 6))
                    stream.write(json.dumps(row, ensure_ascii=True) + "\n")
                stream.flush()
                completed = offset + len(batch)
                if completed == len(pending) or completed % (20 * batch_size) == 0:
                    print(f"{name}: {completed}/{len(pending)} in {time.perf_counter() - started:.1f}s", flush=True)
            del model, processor
            if device == "cuda":
                torch.cuda.empty_cache()
