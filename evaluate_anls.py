import json
import argparse
from nltk.metrics.distance import edit_distance


def normalize(text):
    if text is None:
        return ""

    text = str(text).lower().strip()
    text = " ".join(text.split())   # bỏ khoảng trắng thừa

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


def evaluate(data):
    scores = []

    for sample in data:
        pred = sample["prediction"]
        gt = sample["ground_truth"]

        score = anls(pred, gt)
        scores.append(score)

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)

    args = parser.parse_args()

    with open(args.pred, encoding="utf-8") as f:
        data = json.load(f)

    scores = evaluate(data)

    print("=" * 50)
    print(f"Samples : {len(scores)}")
    print(f"ANLS    : {sum(scores)/len(scores)*100:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()