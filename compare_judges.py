"""
Compare accuracy across judge models.
Usage: python compare_judges.py --judge-dir judge/ --judges gpt-4o gpt-4o-mini gemini-2.5-flash-lite gemini-3.1-flash-lite
"""
import argparse
import json
import os
from pathlib import Path


def calc_acc(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    correct = sum(
        1 for v in data["results"].values()
        if v["Judgement"].lower().strip() == "correct"
    )
    total = len(data["results"])
    invalid = total - correct - sum(
        1 for v in data["results"].values()
        if v["Judgement"].lower().strip() == "incorrect"
    )
    return {"correct": correct, "total": total, "invalid": invalid,
            "accuracy": correct / total * 100 if total else 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-dir", default="judge")
    parser.add_argument("--judges", nargs="+",
                        default=["gpt-4o", "gpt-4o-mini",
                                 "gemini-2.5-flash-lite", "gemini-3.1-flash-lite"])
    parser.add_argument("--model", default="desta2", help="Inference model prefix")
    args = parser.parse_args()

    judge_dir = Path(args.judge_dir)
    tracks = ["Gender", "Language", "Emotion", "Animal"]
    subtracks = ["single", "multi"]

    # collect all results
    results = {}  # results[(track, subtrack)][judge] = acc_dict
    for track in tracks:
        for sub in subtracks:
            key = f"{track}/{sub}"
            results[key] = {}
            for judge in args.judges:
                fname = judge_dir / judge / f"{args.model}_{track}_{sub}_judgements.json"
                if fname.exists():
                    results[key][judge] = calc_acc(str(fname))
                else:
                    results[key][judge] = None

    # print table
    col_w = 12
    header = f"{'Track/Sub':<22}" + "".join(f"{j[:col_w]:>{col_w}}" for j in args.judges)
    print(header)
    print("-" * len(header))

    all_correct = {j: 0 for j in args.judges}
    all_total = {j: 0 for j in args.judges}

    for key, judge_results in results.items():
        row = f"{key:<22}"
        for judge in args.judges:
            r = judge_results.get(judge)
            if r:
                row += f"{r['accuracy']:>{col_w}.2f}"
                all_correct[judge] += r["correct"]
                all_total[judge] += r["total"]
            else:
                row += f"{'N/A':>{col_w}}"
        print(row)

    print("-" * len(header))
    overall_row = f"{'Overall':22}"
    for judge in args.judges:
        if all_total[judge]:
            acc = all_correct[judge] / all_total[judge] * 100
            overall_row += f"{acc:>{col_w}.2f}"
        else:
            overall_row += f"{'N/A':>{col_w}}"
    print(overall_row)

    # also save to JSON
    out = {}
    for key, judge_results in results.items():
        out[key] = {j: (r["accuracy"] if r else None) for j, r in judge_results.items()}
    out["overall"] = {
        j: (all_correct[j] / all_total[j] * 100 if all_total[j] else None)
        for j in args.judges
    }
    out_path = "judge_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
