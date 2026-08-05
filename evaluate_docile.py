#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred",
        default="results/docile_predictions.json",
    )
    args = parser.parse_args()

    data = json.load(open(args.pred, encoding="utf-8"))

    y_true = [x["ground_truth"].strip() for x in data]
    y_pred = [x["prediction"].strip() for x in data]

    print("=" * 80)
    print("Overall")
    print("=" * 80)

    print(f"Samples      : {len(y_true)}")
    print(f"Accuracy     : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1     : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Micro F1     : {f1_score(y_true, y_pred, average='micro', zero_division=0):.4f}")

    



    stats = defaultdict(lambda: [0, 0])

    for gt, pred in zip(y_true, y_pred):
        stats[gt][1] += 1
        if gt == pred:
            stats[gt][0] += 1

    for field in sorted(stats):

        correct, total = stats[field]

     
    labels = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    print()
    print("=" * 80)
    print("Confusion Matrix")
    print("=" * 80)

if __name__ == "__main__":
    main()