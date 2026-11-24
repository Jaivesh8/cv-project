import argparse
import csv
import json
from pathlib import Path

from .data import CACHE, PROTOCOL_PATH, make_probes


def main():
    parser = argparse.ArgumentParser(description="Render logged error examples from cached COCO images")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("submission/results/hallucination_gallery.png"))
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps

    config = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    probes, _, _ = make_probes(config, args.pilot)
    lookup = {probe["probe_id"]: probe for probe in probes}
    with (args.results / "gallery_cases.csv").open(encoding="utf-8", newline="") as stream:
        cases = list(csv.DictReader(stream))
    models = list(config["models"])
    kinds = {"false_yes": "False affirmation", "false_no": "Missed object", "wording_flip": "Wording flip"}
    fig, axes = plt.subplots(len(models), len(kinds), figsize=(12.0, 9.5), layout="constrained")
    for row, model in enumerate(models):
        for col, (kind, label) in enumerate(kinds.items()):
            ax = axes[row, col]
            ax.set_axis_off()
            ax.set_title(f"{model} · {label}", fontsize=11, weight="bold")
            case = next((c for c in cases if c["model"] == model and c["type"] == kind), None)
            if case is None:
                ax.text(.5, .5, "No matching response", ha="center", va="center", transform=ax.transAxes)
                continue
            probe = lookup[case["probe_id"]]
            with Image.open(CACHE / "images" / probe["image"]) as image:
                preview = ImageOps.exif_transpose(image).convert("RGB")
                preview.thumbnail((480, 265), Image.Resampling.LANCZOS)
            ax.imshow(preview)
            ax.text(.5, -.025, f'{case["probe_id"]} · target: {probe["target"]} · label: {case["label"]}\n'
                               f'W0: {case["W0"]}   W1: {case["W1"]}   W2: {case["W2"]}',
                    ha="center", va="top", fontsize=9, transform=ax.transAxes)
    fig.suptitle("COCO existence probes · first logged case by error type", fontsize=15, weight="bold")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight", pad_inches=.2)
    plt.close(fig)


if __name__ == "__main__":
    main()
