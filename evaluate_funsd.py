#!/usr/bin/env python3

import json
import argparse
from collections import Counter
from nltk.metrics.distance import edit_distance


def normalize(text):
    if text is None:
        return ""

    text = str(text).lower().strip()
    text = " ".join(text.split())
    return text


def anls(pred, gt, threshold=0.5):
    pred = normalize(pred)
    gt = normalize(gt)

    if pred == gt:
        return 1.0

    if len(pred) == 0 or len(gt) == 0:
        return 0.0

    dist = edit_distance(pred, gt)
    nl = dist / max(len(pred), len(gt))

    if nl >= threshold:
        return 0.0

    return 1.0 - nl


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    args = parser.parse_args()

    with open(args.pred, encoding="utf-8") as f:
        data = json.load(f)

    # -------------------------------------------------------
    # Đếm số lần mỗi (document, question) xuất hiện
    # -------------------------------------------------------
    cnt = Counter()

    for x in data:
        key = (x["document"], x["question"])
        cnt[key] += 1

    amb_scores = []
    non_scores = []

    for x in data:

        score = anls(
            x["prediction"],
            x["ground_truth"]
        )

        key = (x["document"], x["question"])

        if cnt[key] > 1:
            amb_scores.append(score)
        else:
            non_scores.append(score)

    all_scores = amb_scores + non_scores

    print("=" * 60)
    print(f"Total QA        : {len(all_scores)}")
    print(f"Ambiguous QA    : {len(amb_scores)}")
    print(f"Non-ambiguous QA: {len(non_scores)}")
    print("=" * 60)

    if amb_scores:
        print(f"ANLS (ambiguous)     : {100*sum(amb_scores)/len(amb_scores):.2f}")

    if non_scores:
        print(f"ANLS (non-ambiguous) : {100*sum(non_scores)/len(non_scores):.2f}")

    print("-" * 60)
    print(f"Overall              : {100*sum(all_scores)/len(all_scores):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()