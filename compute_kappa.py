"""
Compute Cohen's Kappa between each judge and GPT-4o (reference),
per track/subtrack, then plot a 16:9 grouped bar chart.
"""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

TRACKS = ["Gender", "Language", "Emotion", "Animal"]
SUBTRACKS = ["single", "multi"]
JUDGES = ["gpt-4o", "gpt-4o-mini", "gemini-2.5-flash-lite", "gemini-3.1-flash-lite"]
REFERENCE = "gpt-4o"

COLORS = {
    "gpt-4o":                  "#2563EB",  # blue
    "gpt-4o-mini":             "#16A34A",  # green
    "gemini-2.5-flash-lite":   "#D97706",  # amber
    "gemini-3.1-flash-lite":   "#DC2626",  # red
}

LABELS = {
    "gpt-4o":                  "GPT-4o (ref)",
    "gpt-4o-mini":             "GPT-4o-mini",
    "gemini-2.5-flash-lite":   "Gemini 2.5 Flash-Lite",
    "gemini-3.1-flash-lite":   "Gemini 3.1 Flash-Lite",
}


def load_judgements(judge_dir: Path, judge: str, track: str, sub: str):
    """Return dict of {audio_path: 'correct'|'incorrect'}, errors excluded."""
    path = judge_dir / judge / f"desta2_{track}_{sub}_judgements.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return {
        k: v["Judgement"].strip().lower()
        for k, v in data["results"].items()
        if v["Judgement"].strip().lower() in ("correct", "incorrect")
    }


def cohen_kappa(labels_a: list, labels_b: list) -> float:
    """Compute Cohen's Kappa for binary (correct/incorrect) labels."""
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return float("nan")

    # observed agreement
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n

    # expected agreement by chance
    cats = ["correct", "incorrect"]
    pe = 0.0
    for cat in cats:
        pa = labels_a.count(cat) / n
        pb = labels_b.count(cat) / n
        pe += pa * pb

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-dir", default="judge")
    parser.add_argument("--model", default="desta2")
    parser.add_argument("--out", default="kappa_comparison.png")
    args = parser.parse_args()

    judge_dir = Path(args.judge_dir)

    # ── collect kappas ─────────────────────────────────────────────────────────
    track_labels = []
    kappa_table = {j: [] for j in JUDGES}

    for track in TRACKS:
        for sub in SUBTRACKS:
            label = f"{track}\n{sub}"
            track_labels.append(label)

            ref = load_judgements(judge_dir, REFERENCE, track, sub)
            if ref is None:
                for j in JUDGES:
                    kappa_table[j].append(float("nan"))
                continue

            ref_keys = sorted(ref.keys())
            ref_vals = [ref[k] for k in ref_keys]

            for judge in JUDGES:
                if judge == REFERENCE:
                    kappa_table[judge].append(1.0)
                    continue
                other = load_judgements(judge_dir, judge, track, sub)
                if other is None:
                    kappa_table[judge].append(float("nan"))
                    continue
                # align on shared keys
                shared = [k for k in ref_keys if k in other]
                if len(shared) < len(ref_keys):
                    print(f"  Warning [{judge}] {track}/{sub}: "
                          f"{len(ref_keys)-len(shared)} keys missing, using {len(shared)}")
                a = [ref[k] for k in shared]
                b = [other[k] for k in shared]
                kappa_table[judge].append(cohen_kappa(a, b))

    # ── print table ─────────────────────────────────────────────────────────────
    col = 26
    header = f"{'Track/Sub':<18}" + "".join(f"{LABELS[j]:>{col}}" for j in JUDGES)
    print(header)
    print("-" * len(header))
    for i, tl in enumerate(track_labels):
        row = f"{tl.replace(chr(10), '/'):18}"
        for j in JUDGES:
            v = kappa_table[j][i]
            row += f"{v:>{col}.4f}" if not np.isnan(v) else f"{'N/A':>{col}}"
        print(row)

    # ── save JSON ───────────────────────────────────────────────────────────────
    out_json = {
        tl.replace("\n", "/"): {j: kappa_table[j][i] for j in JUDGES}
        for i, tl in enumerate(track_labels)
    }
    with open("kappa_results.json", "w") as f:
        json.dump(out_json, f, indent=2)
    print("\nSaved: kappa_results.json")

    # ── plot ────────────────────────────────────────────────────────────────────
    n_tracks = len(track_labels)
    n_judges = len(JUDGES)
    bar_w = 0.18
    x = np.arange(n_tracks)

    fig, ax = plt.subplots(figsize=(16, 9))

    offsets = np.linspace(-(n_judges - 1) / 2, (n_judges - 1) / 2, n_judges) * bar_w

    for idx, judge in enumerate(JUDGES):
        vals = kappa_table[judge]
        bars = ax.bar(
            x + offsets[idx],
            [v if not np.isnan(v) else 0 for v in vals],
            width=bar_w,
            color=COLORS[judge],
            alpha=0.88,
            label=LABELS[judge],
            zorder=3,
        )
        # annotate value on each bar
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5, color="#333333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(track_labels, fontsize=11)
    ax.set_ylabel("Cohen's Kappa (vs GPT-4o)", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "Cohen's Kappa: Agreement with GPT-4o per Track × Subtrack\n"
        "(DeSTA2 predictions, 4 Judge Models)",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
