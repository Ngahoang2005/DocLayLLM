#!/usr/bin/env python3

"""
Build DocILE instructions following VDInstruct.

Input
------
dataset/docile/val/
    annotations/
    ocr_doclayllm/
    match/

Output
------
dataset/docile/val/instruction/
"""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent

ANN_DIR = ROOT / "dataset/docile/val/annotations"
OCR_DIR = ROOT / "dataset/docile/val/ocr_doclayllm"
MATCH_DIR = ROOT / "dataset/docile/val/match"

OUT_DIR = ROOT / "dataset/docile/val/instruction"

import random

CATEGORIES = [
    "account_num",
    "amount_due",
    "amount_paid",
    "amount_total_grass",
    "amount_total_net",
    "amount_total_tax",
    "bank_num",
    "bic",
    "currency_code_amount_due",
    "customer_billing_address",
    "customer_billing_name",
    "customer_delivery_address",
    "customer_delivery_name",
    "customer_id",
    "customer_order_id",
    "customer_other_address",
    "customer_other_name",
    "customer_registration_id",
    "customer_tax_id",
    "date_due",
    "date_issue",
    "document_id",
    "iban",
    "order_id",
    "payment_reference",
    "payment_terms",
    "tax_detail_gross",
    "tax_detail_net",
    "tax_detail_rate",
    "tax_detail_tax",
    "vendor_address",
    "vendor_email",
    "vendor_name",
    "vendor_order_id",
    "vendor_registration_id",
    "vendor_tax_id",
    "total",
]
CATEGORY_TEXT = ", ".join(f'"{x}"' for x in CATEGORIES)

PROMPTS = [
    f'There are 36 categories for selection: {CATEGORY_TEXT}. '
    'Please output the category corresponding to the text "<key>".',

    f'Options: {CATEGORY_TEXT}. '
    'Please select the category associated with the text "<key>" in the given document.',

    f'Please tell me the category of the text "<key>" to select from the following classes: {CATEGORY_TEXT}.',

    f'Categories: {CATEGORY_TEXT}. '
    'Kindly provide me with the category of the text "<key>".',

    f'The document contains 36 key categories: {CATEGORY_TEXT}. '
    'Kindly identify the category related to the text "<key>" mentioned in the provided document.',
]


def build_prompt(text):
    prompt = random.choice(PROMPTS)
    return prompt.replace("<key>", text)

def main():

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    docs = sorted(MATCH_DIR.glob("*.json"))

    print(f"{len(docs)} documents")

    for idx, file in enumerate(docs, 1):

        matches = json.load(open(file, encoding="utf-8"))

        samples = []

        for m in matches:

            samples.append(
                {
                    "page": m["page"],

                    "instruction": build_prompt(
                        m["text"]
                    ),

                    "answer": m["fieldtype"],

                    "fieldtype": m["fieldtype"],

                    "text": m["text"],

                    "start": m["start"],
                    "end": m["end"],

                    "score": m["score"],
                }
            )

        with open(
            OUT_DIR / file.name,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                samples,
                f,
                ensure_ascii=False,
                indent=2,
            )

        if idx % 50 == 0 or idx == len(docs):

            print(f"{idx}/{len(docs)}")

    print("Done.")


if __name__ == "__main__":
    main()