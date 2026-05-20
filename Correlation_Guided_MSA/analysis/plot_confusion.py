import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def load_columns(prediction_path, task):
    label_key = "label_2" if task == "binary" else "label_7"
    pred_key = "pred_2" if task == "binary" else "pred_7"
    labels = []
    preds = []
    with open(prediction_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row[label_key]))
            preds.append(int(row[pred_key]))
    return np.asarray(labels), np.asarray(preds)


def get_display_labels(task):
    if task == "binary":
        return ["neg", "non-neg"]
    return [str(x) for x in range(-3, 4)]


def plot_single(ax, labels, preds, display_labels, normalize, title):
    cm = confusion_matrix(
        labels,
        preds,
        labels=list(range(len(display_labels))),
        normalize="true" if normalize else None,
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f" if normalize else "d")
    ax.set_title(title)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True, help="Current model predictions.csv")
    parser.add_argument("--baseline-predictions", type=str, default="", help="Optional baseline predictions.csv")
    parser.add_argument("--task", type=str, choices=["binary", "seven"], default="seven")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    display_labels = get_display_labels(args.task)
    current_labels, current_preds = load_columns(args.predictions, args.task)
    if args.baseline_predictions:
        baseline_labels, baseline_preds = load_columns(args.baseline_predictions, args.task)
        if len(baseline_labels) != len(current_labels) or not np.array_equal(baseline_labels, current_labels):
            raise ValueError("Baseline and current predictions must use the same test-set ordering and labels.")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plot_single(axes[0], baseline_labels, baseline_preds, display_labels, args.normalize, "Baseline")
        plot_single(axes[1], current_labels, current_preds, display_labels, args.normalize, "Ours")
        fig.suptitle(f"{args.task} confusion matrix")
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_single(ax, current_labels, current_preds, display_labels, args.normalize, f"{args.task} confusion matrix")

    fig.tight_layout()

    default_name = f"confusion_compare_{args.task}.png" if args.baseline_predictions else f"confusion_{args.task}.png"
    output = Path(args.output) if args.output else Path(args.predictions).with_name(default_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"saved confusion figure to {output}")


if __name__ == "__main__":
    main()
