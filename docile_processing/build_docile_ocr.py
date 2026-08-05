#!/usr/bin/env python3
"""
Convert DocILE OCR to DocLayLLM OCR format.
Validation split only.
"""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent

OCR_DIR = ROOT / "dataset" / "docile" / "val" / "ocr"
OUT_DIR = ROOT / "dataset" / "docile" / "val" / "ocr_doclayllm"

from statistics import median


def row_clustering(records):
    """
    Reading order:
        1. Cluster by y-center
        2. Sort inside each row by x
    """

    if not records:
        return records

    heights = [r["box"][3] - r["box"][1] for r in records]
    med_h = median(heights)

    # giống heuristic đã dùng cho FUNSD
    threshold = max(8, med_h * 0.6)

    # sort theo tâm y trước
    records = sorted(
        records,
        key=lambda r: (
            (r["box"][1] + r["box"][3]) / 2,
            r["box"][0],
        ),
    )

    rows = []

    for rec in records:

        cy = (rec["box"][1] + rec["box"][3]) / 2

        assigned = False

        for row in rows:

            if abs(cy - row["cy"]) <= threshold:

                row["items"].append(rec)

                n = len(row["items"])

                row["cy"] = (row["cy"] * (n - 1) + cy) / n

                assigned = True
                break

        if not assigned:

            rows.append(
                {
                    "cy": cy,
                    "items": [rec],
                }
            )

    output = []

    rows.sort(key=lambda r: r["cy"])

    for row in rows:

        row["items"].sort(key=lambda r: r["box"][0])

        output.extend(row["items"])

    return output
def geometry_to_box(geometry, page_width, page_height):
    (x1, y1), (x2, y2) = geometry

    return [
        int(round(x1 * page_width)),
        int(round(y1 * page_height)),
        int(round(x2 * page_width)),
        int(round(y2 * page_height)),
    ]


def union_boxes(boxes):
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def convert_file(path):

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    records = []

    for page_index, page in enumerate(data["pages"]):

        # DocILE stores (height, width)
        page_height, page_width = page["dimensions"]

        for block in page.get("blocks", []):

            for line in block.get("lines", []):

                words = line.get("words", [])

                if len(words) == 0:
                    continue

                texts = []
                boxes = []

                for word in words:

                    txt = word["value"].strip()

                    if txt == "":
                        continue

                    texts.append(txt)

                    boxes.append(
                        geometry_to_box(
                            word["geometry"],
                            page_width,
                            page_height,
                        )
                    )

                if len(texts) == 0:
                    continue

                records.append(
                    {
                        "text": " ".join(texts),
                        "box": union_boxes(boxes),
                        "page": page_index,
                    }
                )

    records = row_clustering(records)
    return records


def main():

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(OCR_DIR.glob("*.json"))

    print(f"Found {len(files)} OCR files")

    for i, file in enumerate(files, 1):

        out = OUT_DIR / file.name

        records = convert_file(file)

        with open(out, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        if i % 50 == 0 or i == len(files):
            print(f"{i}/{len(files)}")

    print("Done.")


if __name__ == "__main__":
    main()