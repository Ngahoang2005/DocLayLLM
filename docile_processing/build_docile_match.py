#!/usr/bin/env python3

from pathlib import Path
import json
import re
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parent.parent

ANN_DIR = ROOT / "dataset/docile/val/annotations"
OCR_DIR = ROOT / "dataset/docile/val/ocr_doclayllm"
OUT_DIR = ROOT / "dataset/docile/val/match"


def normalize(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_windows(lines, max_window=5):
    """
    Build windows of consecutive OCR lines.
    """

    windows = []

    n = len(lines)

    for start in range(n):

        text = ""

        for end in range(start, min(start + max_window, n)):

            if text:
                text += " "

            text += lines[end]["text"]

            windows.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )

    return windows


def similarity(gt, pred):

    gt = normalize(gt)
    pred = normalize(pred)

    ratio = fuzz.ratio(gt, pred)
    partial = fuzz.partial_ratio(gt, pred)

    # ưu tiên chuỗi giống hoàn toàn nhưng vẫn cho phép OCR tách dòng
    return max(ratio, partial)


def match_document(docid):

    ann = json.load(open(ANN_DIR / f"{docid}.json", encoding="utf-8"))
    ocr = json.load(open(OCR_DIR / f"{docid}.json", encoding="utf-8"))

    pages = {}

    for line in ocr:
        pages.setdefault(line["page"], []).append(line)

    results = []

    for field in ann["field_extractions"]:

        page = field["page"]
        gt_text = field["text"]

        page_lines = pages.get(page, [])

        windows = build_windows(page_lines, max_window=5)

        best = None
        best_score = -1

        for w in windows:

            score = similarity(gt_text, w["text"])

            # penalty cửa sổ dài
            window_len = w["end"] - w["start"] + 1
            score -= (window_len - 1) * 5

            if score > best_score:
                best_score = score
                best = w

        results.append(
            {
                "fieldtype": field["fieldtype"],
                "page": page,
                "text": gt_text,
                "score": round(best_score, 2),
                "matched_text": best["text"],
                "start": best["start"],
                "end": best["end"],
            }
        )

    return results


def main():

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    docs = sorted(ANN_DIR.glob("*.json"))

    for i, p in enumerate(docs, 1):

        result = match_document(p.stem)

        with open(OUT_DIR / p.name, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        if i % 50 == 0 or i == len(docs):
            print(f"{i}/{len(docs)}")


if __name__ == "__main__":
    main()