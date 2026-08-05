#!/usr/bin/env python3
"""Build unambiguous FUNSD visual-information-extraction QA pairs.

LayoutLLM constructs FUNSD QA pairs from direct question--answer links and
filters a pair when either endpoint links to multiple entities.  This script
implements that rule using the complete, undirected FUNSD relation graph.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distro import name


SPLITS = ("training_data", "testing_data")


@dataclass(frozen=True)
class SplitStats:
    documents: int = 0
    qa_pairs: int = 0
    filtered_pairs: int = 0
import re

def is_valid_question(text):
    if text is None:
        return False

    text = text.strip()
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text)

    # ":"  "."  "---" ...
    if cleaned == "":
        return False

    return True

def validate_entity(entity: Any, index: int, source: Path) -> dict[str, Any]:
    """Validate the FUNSD fields used for relation and QA construction."""
    if not isinstance(entity, dict):
        raise ValueError(f"{source}: form[{index}] is not an object")
    if not isinstance(entity.get("id"), int):
        raise ValueError(f"{source}: form[{index}].id is not an integer")
    if not isinstance(entity.get("label"), str):
        raise ValueError(f"{source}: form[{index}].label is not a string")
    if not isinstance(entity.get("text"), str):
        raise ValueError(f"{source}: form[{index}].text is not a string")
    if not isinstance(entity.get("linking"), list):
        raise ValueError(f"{source}: form[{index}].linking is not a list")
    return entity


def load_entities(source: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Load a FUNSD annotation and retain its stored entity order."""
    with source.open(encoding="utf-8") as handle:
        annotation = json.load(handle)
    if not isinstance(annotation, dict) or not isinstance(annotation.get("form"), list):
        raise ValueError(f"{source}: expected an annotation object with a 'form' list")

    entities = [validate_entity(entity, index, source) for index, entity in enumerate(annotation["form"])]
    entities_by_id = {entity["id"]: entity for entity in entities}
    if len(entities_by_id) != len(entities):
        raise ValueError(f"{source}: duplicate form entity IDs")
    return entities, entities_by_id


def relation_graph(
    entities: list[dict[str, Any]], entities_by_id: dict[int, dict[str, Any]], source: Path
) -> tuple[set[tuple[int, int]], dict[int, set[int]]]:
    """Create unique undirected FUNSD edges and all-entity adjacency sets."""
    edges: set[tuple[int, int]] = set()
    adjacency: dict[int, set[int]] = defaultdict(set)

    for entity in entities:
        for link_index, pair in enumerate(entity["linking"]):
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(not isinstance(entity_id, int) for entity_id in pair)
            ):
                raise ValueError(
                    f"{source}: entity {entity['id']} linking[{link_index}] must be two integer IDs"
                )
            left, right = pair
            if left not in entities_by_id or right not in entities_by_id:
                raise ValueError(f"{source}: link {pair} refers to an unknown entity ID")
            if left == right:
                continue

            edge = (min(left, right), max(left, right))
            edges.add(edge)

    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    return edges, adjacency


def qa_records(source: Path) -> tuple[list[dict[str, Any]], int]:
    """Return ordered, unambiguous question--answer pairs and filter count."""
    entities, entities_by_id = load_entities(source)
    edges, adjacency = relation_graph(entities, entities_by_id, source)
    entity_order = {entity["id"]: index for index, entity in enumerate(entities)}

    direct_qa_edges: list[tuple[int, int]] = []
    for left, right in edges:
        left_label = entities_by_id[left]["label"]
        right_label = entities_by_id[right]["label"]
        if left_label == "question" and right_label == "answer":
            direct_qa_edges.append((left, right))
        elif left_label == "answer" and right_label == "question":
            direct_qa_edges.append((right, left))

    # The question entity determines primary output order; answer order is the
    # annotation order for completeness before ambiguous pairs are rejected.
    direct_qa_edges.sort(key=lambda pair: (entity_order[pair[0]], entity_order[pair[1]]))

    records: list[dict[str, Any]] = []
    filtered_pairs = 0
    for question_id, answer_id in direct_qa_edges:
        if len(adjacency[question_id]) > 1 or len(adjacency[answer_id]) > 1:
            filtered_pairs += 1
            continue

        question_text = entities_by_id[question_id]["text"]
        if not is_valid_question(question_text):
            continue
        answer_text = entities_by_id[answer_id]["text"]
        if not question_text.strip() or not answer_text.strip():
            continue

        records.append(
            {
                "question": f'What is the "{question_text}" in the document?',
                "answer": answer_text,
                "question_id": question_id,
                "answer_id": answer_id,
            }
        )

    return records, filtered_pairs


def build_split(dataset_root: Path, split: str) -> SplitStats:
    """Write one QA JSON file for every annotation in a FUNSD split."""
    annotation_dir = dataset_root / split / "annotations"
    output_dir = dataset_root / split / "qa"
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory does not exist: {annotation_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = SplitStats()
    for source in sorted(annotation_dir.glob("*.json")):
        records, filtered_pairs = qa_records(source)
        with (output_dir / source.name).open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        stats = SplitStats(
            documents=stats.documents + 1,
            qa_pairs=stats.qa_pairs + len(records),
            filtered_pairs=stats.filtered_pairs + filtered_pairs,
        )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert FUNSD annotations into unambiguous LayoutLLM-style QA pairs."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset",
        help="FUNSD dataset directory (default: %(default)s)",
    )
    args = parser.parse_args()

    for split in SPLITS:
        stats = build_split(args.dataset_root, split)
        name = split[:-5] if split.endswith("_data") else split
        print(f"{name}:")
        print(f"    documents: {stats.documents}")
        print(f"    qa_pairs: {stats.qa_pairs}")
        print(f"    filtered_pairs: {stats.filtered_pairs}")


if __name__ == "__main__":
    main()
