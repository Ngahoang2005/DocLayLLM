#!/usr/bin/env python3
"""Build DocLayLLM OCR inputs from FUNSD entity annotations.

The demo consumes an ordered JSON array of ``{"text": ..., "box": ...}``
records.  FUNSD's ``form`` entries already provide that entity-level OCR
representation; their nested ``words`` are not needed by the demo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SPLITS = ("training_data", "testing_data")


def ocr_records(annotation: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    """Return non-empty FUNSD form entries in their stored order."""
    form = annotation.get("form")
    if not isinstance(form, list):
        raise ValueError(f"{source}: expected a list in the 'form' field")

    records: list[dict[str, Any]] = []
    for index, entity in enumerate(form):
        if not isinstance(entity, dict):
            raise ValueError(f"{source}: form[{index}] is not an object")

        text = entity.get("text")
        if not isinstance(text, str):
            raise ValueError(f"{source}: form[{index}].text is not a string")
        if not text.strip():
            continue

        box = entity.get("box")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(not isinstance(coordinate, (int, float)) for coordinate in box)
        ):
            raise ValueError(f"{source}: form[{index}].box must be four numbers")

        # Keep FUNSD's original text and pixel coordinates unchanged.
        records.append({"box": box, "text": text})

    return records


def build_split(dataset_root: Path, split: str) -> tuple[int, int]:
    """Convert every annotation in one FUNSD split and return file/record counts."""
    annotation_dir = dataset_root / split / "annotations"
    output_dir = dataset_root / split / "ocr"
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory does not exist: {annotation_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    document_count = 0
    record_count = 0
    for annotation_path in sorted(annotation_dir.glob("*.json")):
        with annotation_path.open(encoding="utf-8") as handle:
            annotation = json.load(handle)
        if not isinstance(annotation, dict):
            raise ValueError(f"{annotation_path}: annotation root is not an object")

        records = ocr_records(annotation, annotation_path)
        output_path = output_dir / annotation_path.name
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        document_count += 1
        record_count += len(records)

    return document_count, record_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert FUNSD form annotations into DocLayLLM OCR JSON files."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset",
        help="FUNSD dataset directory (default: %(default)s)",
    )
    args = parser.parse_args()

    for split in SPLITS:
        documents, records = build_split(args.dataset_root, split)
        print(f"{split}: wrote {documents} OCR files with {records} text segments")


if __name__ == "__main__":
    main()
