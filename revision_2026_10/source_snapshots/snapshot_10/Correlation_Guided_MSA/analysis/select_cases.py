import argparse
import csv
from pathlib import Path


def load_cases(case_path):
    rows = []
    with open(case_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label"] = float(row["label"])
            row["pred"] = float(row["pred"])
            row["r_t"] = float(row["r_t"]) if row["r_t"] else float("nan")
            row["r_a"] = float(row["r_a"]) if row["r_a"] else float("nan")
            row["r_v"] = float(row["r_v"]) if row["r_v"] else float("nan")
            row["abs_error"] = abs(row["pred"] - row["label"])
            reliabilities = [row["r_t"], row["r_a"], row["r_v"]]
            valid = [x for x in reliabilities if x == x]
            row["mean_r"] = sum(valid) / len(valid) if valid else float("nan")
            rows.append(row)
    return rows


def load_case_map(case_path):
    return {row["sample_id"]: row for row in load_cases(case_path)}


def write_rows(output, rows, fieldnames):
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved selected cases to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=str, required=True, help="Current model cases.csv")
    parser.add_argument("--baseline-cases", type=str, default="", help="Optional baseline cases.csv")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hard", "easy", "low_reliability", "high_reliability", "improved", "regressed"],
        default="hard",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    if args.mode in {"improved", "regressed"}:
        if not args.baseline_cases:
            raise ValueError("--baseline-cases is required for improved/regressed modes.")
        ours = load_case_map(args.cases)
        baseline = load_case_map(args.baseline_cases)
        rows = []
        for sample_id, row in ours.items():
            if sample_id not in baseline:
                continue
            base = baseline[sample_id]
            delta_error = base["abs_error"] - row["abs_error"]
            merged = {
                "sample_id": sample_id,
                "text": row["text"],
                "label": row["label"],
                "baseline_pred": base["pred"],
                "ours_pred": row["pred"],
                "baseline_abs_error": base["abs_error"],
                "ours_abs_error": row["abs_error"],
                "delta_error": delta_error,
                "r_t": row["r_t"],
                "r_a": row["r_a"],
                "r_v": row["r_v"],
                "mean_r": row["mean_r"],
            }
            rows.append(merged)
        reverse = args.mode == "improved"
        ranked = sorted(rows, key=lambda x: x["delta_error"], reverse=reverse)[: args.top_k]
        output = Path(args.output) if args.output else Path(args.cases).with_name(f"selected_{args.mode}_top{args.top_k}.csv")
        write_rows(
            output,
            ranked,
            [
                "sample_id",
                "text",
                "label",
                "baseline_pred",
                "ours_pred",
                "baseline_abs_error",
                "ours_abs_error",
                "delta_error",
                "r_t",
                "r_a",
                "r_v",
                "mean_r",
            ],
        )
        return

    rows = load_cases(args.cases)
    reverse = True
    key = "abs_error"
    if args.mode == "easy":
        reverse = False
    elif args.mode == "low_reliability":
        key = "mean_r"
        reverse = False
    elif args.mode == "high_reliability":
        key = "mean_r"

    ranked = sorted(rows, key=lambda x: x[key], reverse=reverse)[: args.top_k]
    output = Path(args.output) if args.output else Path(args.cases).with_name(f"selected_{args.mode}_top{args.top_k}.csv")
    write_rows(
        output,
        ranked,
        [
            "sample_id",
            "text",
            "label",
            "pred",
            "abs_error",
            "r_t",
            "r_a",
            "r_v",
            "mean_r",
        ],
    )


if __name__ == "__main__":
    main()
