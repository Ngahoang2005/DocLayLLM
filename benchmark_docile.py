#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

from infer_demo import load_model, predict


def resolve_image(image_dir: Path, docid: str, page: int):

    single = image_dir / f"{docid}.png"
    if single.exists():
        return single

    multi = image_dir / f"{docid}_{page}.png"
    if multi.exists():
        return multi

    raise FileNotFoundError(
        f"Cannot find image for {docid}, page={page}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        required=True,
    )

    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("dataset/docile/val"),
    )

    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("results/docile_predictions.json"),
    )

    parser.add_argument(
        "--doc",
        default=None,
        help="Run only one document id.",
    )

    args = parser.parse_args()

    image_dir = args.dataset_dir / "images"
    ocr_dir = args.dataset_dir / "ocr_doclayllm"
    instruction_dir = args.dataset_dir / "instruction"

    docs = sorted(instruction_dir.glob("*.json"))

    if args.doc is not None:
        docs = [instruction_dir / f"{args.doc}.json"]

    print(f"{len(docs)} documents")

    model, tokenizer, generator_config = load_model(args.model_dir)

    predictions = []

    start = time.perf_counter()

    for doc_idx, ins_file in enumerate(docs, 1):

        docid = ins_file.stem

        print(f"[{doc_idx}/{len(docs)}] {docid}")

        samples = json.load(open(ins_file, encoding="utf-8"))

        ocr_file = ocr_dir / f"{docid}.json"

        for i, sample in enumerate(samples, 1):

            page = sample["page"]

            image_path = resolve_image(
                image_dir,
                docid,
                page,
            )

            print(
                f"[{doc_idx}/{len(docs)}] "
                f"{docid} "
                f"sample={i}/{len(samples)} "
                f"page={page}",
                flush=True,
            )

            pred = predict(
                model=model,
                tokenizer=tokenizer,
                generator_config=generator_config,
                img_dir=str(image_path),
                ocr_dir=str(ocr_file),
                instruction=sample["instruction"],
                page=page,
            )

            predictions.append(
                {
                    "document": docid,
                    "page": page,
                    "fieldtype": sample["fieldtype"],
                    "ground_truth": sample["answer"],
                    "prediction": pred,
                    "text": sample["text"],
                }
            )

        # checkpoint sau mỗi document
        args.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            args.output_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                predictions,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Checkpoint saved ({len(predictions)} samples)")

    elapsed = time.perf_counter() - start

    print()
    print("=" * 80)
    print("Finished")
    print("=" * 80)
    print(f"documents : {len(docs)}")
    print(f"samples   : {len(predictions)}")
    print(f"time      : {elapsed:.2f} sec")


if __name__ == "__main__":
    main()