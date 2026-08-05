#!/usr/bin/env python3

from pathlib import Path
import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent

PDF_DIR = ROOT / "dataset/docile/val/pdfs"
OUT_DIR = ROOT / "dataset/docile/val/images"

DPI = 200


def pdf_to_png(pdf_path, out_dir):

    doc = fitz.open(pdf_path)

    zoom = DPI / 72
    mat = fitz.Matrix(zoom, zoom)

    for page_idx in range(len(doc)):

        pix = doc[page_idx].get_pixmap(
            matrix=mat,
            alpha=False,
        )

        if len(doc) == 1:
            out = out_dir / f"{pdf_path.stem}.png"
        else:
            out = out_dir / f"{pdf_path.stem}_{page_idx}.png"

        pix.save(out)


def main():

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(PDF_DIR.glob("*.pdf"))

    print(f"{len(pdfs)} pdfs")

    for i, pdf in enumerate(pdfs, 1):

        pdf_to_png(pdf, OUT_DIR)

        if i % 50 == 0 or i == len(pdfs):
            print(f"{i}/{len(pdfs)}")

    print("Done")


if __name__ == "__main__":
    main()