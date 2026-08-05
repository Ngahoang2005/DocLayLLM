#!/usr/bin/env python3

"""
Analyze DocILE OCR matching quality.

Output:
- score distribution
- average score
- window length statistics
- worst matched fields
"""

import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
MATCH_DIR = ROOT / "dataset/docile/val/match"

all_scores = []
window_counter = Counter()
bad_cases = []

files = sorted(MATCH_DIR.glob("*.json"))

if len(files) == 0:
    raise RuntimeError("No match files found.")

for file in files:

    matches = json.load(open(file, encoding="utf-8"))

    for m in matches:

        score = m["score"]

        all_scores.append(score)

        window_len = m["end"] - m["start"] + 1
        window_counter[window_len] += 1

        if score < 80:
            bad_cases.append({
                "doc": file.stem,
                "fieldtype": m["fieldtype"],
                "score": score,
                "annotation": m["text"],
                "matched": m["matched_text"],
            })

total = len(all_scores)

print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"documents : {len(files)}")
print(f"fields    : {total}")
print()

print(f"average score : {sum(all_scores)/total:.2f}")
print()

for th in [100,95,90,80,70,60,50]:

    cnt = sum(s >= th for s in all_scores)

    print(f">= {th:3d}: {cnt:6d} ({cnt/total*100:6.2f}%)")

print()

print("=" * 80)
print("WINDOW LENGTH")
print("=" * 80)

for k in sorted(window_counter):

    print(f"{k} line(s): {window_counter[k]}")

print()

print("=" * 80)
print("50 WORST MATCHES")
print("=" * 80)

bad_cases.sort(key=lambda x: x["score"])

for x in bad_cases[:50]:

    print("-" * 60)
    print("doc       :", x["doc"])
    print("fieldtype :", x["fieldtype"])
    print("score     :", x["score"])
    print()
    print("Annotation")
    print(x["annotation"])
    print()
    print("Matched")
    print(x["matched"])
    print()

print("=" * 80)
print(f"Bad cases (<80): {len(bad_cases)}")
print("=" * 80)