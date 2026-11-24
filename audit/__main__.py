import argparse
import json
from pathlib import Path

from .data import PROTOCOL_PATH, digest, make_probes
from .inference import run


def main():
    parser = argparse.ArgumentParser(description="UCS532 P44 paired POPE-style object existence audit")
    parser.add_argument("--pilot", action="store_true", help="Use the separate 12-probe smoke-test split")
    parser.add_argument("--replay", action="store_true", help="Rebuild tables from complete prediction logs without inference")
    parser.add_argument("--offline", action="store_true", help="Use downloaded images and checkpoint cache")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    out = args.out or Path(".cache/pilot" if args.pilot else "results")
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if args.model and args.model not in config["models"]:
        parser.error(f"Unknown checkpoint: {args.model}")
    probes, hashes, manifest = make_probes(config, args.pilot)
    metadata = dict(
        seed=config["seed"], bootstrap_seed=config["bootstrap_seed"], pilot=args.pilot,
        n_probes=len(probes), n_images=len({p["image"] for p in probes}),
        protocol_sha256=digest(PROTOCOL_PATH), manifest_sha256=manifest, annotation_sha256=hashes,
        models=config["models"],
    )
    if args.replay:
        recorded = json.loads((out / "run.json").read_text(encoding="utf-8"))
        if any(recorded.get(key) != value for key, value in metadata.items()):
            raise ValueError("Recorded settings do not match the selected experimental protocol")
        print(json.dumps(recorded), flush=True)
    else:
        run(config, probes, metadata, out, args.device, args.batch_size, args.offline, args.model)
    if args.model:
        return
    from .reporting import analyze
    analyze(config, probes, out)


if __name__ == "__main__":
    main()
