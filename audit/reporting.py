import csv
from collections import defaultdict

import numpy as np

from .inference import load_predictions


def table(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentage(n, d):
    n, d = np.asarray(n), np.asarray(d)
    return 100 * np.divide(n, d, out=np.full(np.broadcast(n, d).shape, np.nan), where=d > 0)


def metrics(positive, answer, idx):
    yes, no = answer == "yes", answer == "no"
    tp, fp = (yes & positive)[idx].sum(axis=-1), (yes & ~positive)[idx].sum(axis=-1)
    fn, tn = (no & positive)[idx].sum(axis=-1), (no & ~positive)[idx].sum(axis=-1)
    p, n = positive[idx].sum(axis=-1), (~positive)[idx].sum(axis=-1)
    return {
        "precision": percentage(tp, tp + fp), "recall": percentage(tp, p),
        "f1": percentage(2 * tp, tp + fp + p), "accuracy": percentage(tp + tn, p + n),
        "false_yes": percentage(fp, n), "false_no": percentage(fn, p),
        "yes_rate": percentage(tp + fp, p + n),
        "invalid": percentage(p + n - tp - fp - tn - fn, p + n),
    }


def estimates(point, samples):
    out = {}
    for name, value in point.items():
        finite = np.asarray(samples[name])
        finite = finite[np.isfinite(finite)]
        lower, upper = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
        out.update({name: round(float(value), 3), f"{name}_low": round(float(lower), 3),
                    f"{name}_high": round(float(upper), 3)})
    return out


def exact_paired_test(changes):
    """Two-sided sign permutation, with both probes from an image moved together."""
    observed = abs(sum(changes))
    counts = {0: 1}
    for change in changes:
        if change:
            next_counts = defaultdict(int)
            for total, count in counts.items():
                next_counts[total - change] += count
                next_counts[total + change] += count
            counts = next_counts
    return sum(count for total, count in counts.items() if abs(total) >= observed) / sum(counts.values())


def analyze(config, probes, out):
    predictions = load_predictions(out / "predictions.jsonl")
    models = list(config["models"])
    arms = [*config["templates"], "blank_W0"]
    expected = {(name, p["probe_id"], arm) for p in probes for name in models for arm in arms}
    if predictions.keys() != expected:
        raise ValueError(f"Log is incomplete or inconsistent: {len(expected - predictions.keys())} missing, "
                         f"{len(predictions.keys() - expected)} extra")
    truth = np.array([p["label"] for p in probes])
    old_truth = np.array([p["original_label"] for p in probes])
    labels = truth == "yes"
    point_idx = np.arange(len(probes))
    # Resample paired positive/negative probes together, within each image's negative strategy.
    by_image = defaultdict(list)
    for index, probe in enumerate(probes):
        by_image[probe["image"]].append(index)
    strata = defaultdict(list)
    for group in by_image.values():
        if len(group) != 2 or labels[group].sum() != 1:
            raise ValueError("Each image must contribute exactly one probe of each label")
        negative = next(i for i in group if not labels[i])
        strata[probes[negative]["split"]].append(group)
    rng = np.random.default_rng(config["bootstrap_seed"])
    bootstrap = np.concatenate([
        np.asarray(groups)[rng.integers(len(groups), size=(config["bootstrap_draws"], len(groups)))]
        .reshape(config["bootstrap_draws"], -1)
        for groups in strata.values()
    ], axis=1)

    scores, ablation, flip_ablation, flips, slices, rescoring, cases, timing = [], [], [], [], [], [], [], []
    for model in models:
        answers = {
            arm: np.array([predictions[model, p["probe_id"], arm]["answer"] for p in probes])
            for arm in arms
        }
        base = answers["W0"]
        for arm, answer in answers.items():
            point, samples = metrics(labels, answer, point_idx), metrics(labels, answer, bootstrap)
            counts = dict(
                n=len(probes), positives=int(labels.sum()), negatives=int((~labels).sum()),
                true_yes=int(((answer == "yes") & labels).sum()),
                false_yes_count=int(((answer == "yes") & ~labels).sum()),
                true_no=int(((answer == "no") & ~labels).sum()),
                false_no_count=int(((answer == "no") & labels).sum()),
                invalid_count=int((answer == "invalid").sum()),
            )
            row = dict(model=model, arm=arm, **counts, **estimates(point, samples))
            if arm in {"W0", "blank_W0"}:
                ablation.append(row)
            if arm in config["templates"]:
                scores.append(row)
                old = metrics(old_truth == "yes", answer, point_idx)
                rescoring.append(dict(model=model, arm=arm, n=len(probes),
                                      relabeled=int((old_truth != truth).sum()),
                                      original_f1=round(float(old["f1"]), 3),
                                      corrected_f1=row["f1"],
                                      original_false_yes=round(float(old["false_yes"]), 3),
                                      corrected_false_yes=row["false_yes"]))

        for split in config["images_per_negative_strategy"]:
            idx = np.array([i for i, p in enumerate(probes) if p["label"] == "no" and p["split"] == split])
            values = base[idx] == "yes"
            samples = values[rng.integers(len(idx), size=(config["bootstrap_draws"], len(idx)))].mean(axis=1) * 100
            summ = estimates({"false_yes": values.mean() * 100}, {"false_yes": samples})
            slices.append(dict(model=model, split=split, n=len(idx), false_yes_count=int(values.sum()), **summ))

        for arm in (a for a in config["templates"] if a != "W0"):
            answer = answers[arm]
            valid = (base != "invalid") & (answer != "invalid")
            toward_yes = (base == "no") & (answer == "yes")
            toward_no = (base == "yes") & (answer == "no")
            gained = (base != truth) & (answer == truth)
            lost = (base == truth) & (answer != truth)
            a, b = int(gained.sum()), int(lost.sum())
            diff = metrics(labels, answer, bootstrap)["false_yes"] - metrics(labels, base, bootstrap)["false_yes"]
            point_diff = metrics(labels, answer, point_idx)["false_yes"] - metrics(labels, base, point_idx)["false_yes"]
            unstable = toward_yes | toward_no
            changes = [int(gained[idx].sum()) - int(lost[idx].sum()) for idx in by_image.values()]
            flips.append(dict(
                model=model, comparison=f"W0 vs {arm}", n=len(probes), valid_pairs=int(valid.sum()),
                no_to_yes=int(toward_yes.sum()), yes_to_no=int(toward_no.sum()),
                corrected=a, introduced=b,
                **estimates(
                    dict(flip_rate=percentage(unstable.sum(), valid.sum()), delta_false_yes=point_diff),
                    dict(flip_rate=percentage(unstable[bootstrap].sum(axis=1), valid[bootstrap].sum(axis=1)),
                         delta_false_yes=diff),
                ),
                p_value=exact_paired_test(changes),
                negative_flips=int((unstable & ~labels).sum()),
                negative_valid_pairs=int((valid & ~labels).sum()),
                positive_flips=int((unstable & labels).sum()),
                positive_valid_pairs=int((valid & labels).sum()),
            ))

        blank = answers["blank_W0"]
        real_point, blank_point = metrics(labels, base, point_idx), metrics(labels, blank, point_idx)
        real_draw, blank_draw = metrics(labels, base, bootstrap), metrics(labels, blank, bootstrap)
        effects = ("accuracy", "recall", "false_yes", "yes_rate")
        flip_ablation.append(dict(model=model, n=len(probes), **estimates(
            {f"real_minus_blank_{k}": real_point[k] - blank_point[k] for k in effects},
            {f"real_minus_blank_{k}": real_draw[k] - blank_draw[k] for k in effects},
        )))

        for kind, mask in {
            "false_yes": (base == "yes") & ~labels,
            "false_no": (base == "no") & labels,
            "wording_flip": (base != "invalid") & np.logical_or.reduce([
                (answers[arm] != base) & (answers[arm] != "invalid") for arm in config["templates"] if arm != "W0"
            ]),
        }.items():
            for i in np.flatnonzero(mask)[:config["gallery_per_type"]]:
                p = probes[i]
                cases.append(dict(model=model, type=kind, probe_id=p["probe_id"],
                                  label=p["label"], **{arm: answers[arm][i] for arm in arms}))
        for arm in arms:
            values = [predictions[model, p["probe_id"], arm]["seconds"] for p in probes]
            timing.append(dict(model=model, arm=arm, responses=len(values), seconds=round(sum(values), 3)))

    adjusted = 0.0
    for rank, row in enumerate(sorted(flips, key=lambda r: r["p_value"])):
        adjusted = max(adjusted, min(1.0, (len(flips) - rank) * row["p_value"]))
        row["holm_p"] = adjusted
    for name, rows in dict(main=scores, ablation=ablation, ablation_effects=flip_ablation,
                           wording=flips, negative_slices=slices,
                           label_rescoring=rescoring, gallery_cases=cases, timing=timing).items():
        table(out / f"{name}.csv", rows)

    markdown = ["| Model | Prompt | Precision | Recall | F1 | False yes | False no | Invalid |",
                "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in scores:
        values = [row["model"], row["arm"]] + [
            f"{row[key]:.2f}%" for key in ("precision", "recall", "f1", "false_yes", "false_no", "invalid")
        ]
        markdown.append("| " + " | ".join(values) + " |")
    (out / "main.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print("\n".join(markdown), flush=True)
    figures(scores, flips, out)


def figures(scores, flips, out):
    from matplotlib import pyplot as plt

    for filename, records, metric, title in (
        ("f1_by_wording", scores, "f1", "Existence-probe F1 by question wording"),
        ("paired_flip_rate", flips, "flip_rate", "Yes/no answer changes on jointly valid probes"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
        for x, row in enumerate(records):
            val, lo, hi = (row.get(k) for k in (metric, f"{metric}_low", f"{metric}_high"))
            if np.isfinite(val) and np.isfinite(lo) and np.isfinite(hi):
                ax.vlines(x, lo, hi, color="#316887", linewidth=2)
                ax.scatter(x, val, color="#174868", s=32, zorder=3)
        ax.set_xticks(range(len(records)),
                      [r["model"] + "\n" + (r["arm"] if "arm" in r else r["comparison"]) for r in records],
                      fontsize=8)
        ax.set(ylabel="Percent (95% bootstrap interval)", title=title, ylim=(-2, 102))
        ax.grid(axis="y", alpha=.25)
        fig.savefig(out / f"{filename}.png", dpi=170)
        plt.close(fig)
