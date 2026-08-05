#!/usr/bin/env python3
"""Run the DocLayLLM demo inference path on the FUNSD test split."""

import argparse
import json
import time
from pathlib import Path

from infer_demo import load_model, predict


def main():
    parser = argparse.ArgumentParser(
        description="Generate DocLayLLM predictions for every FUNSD test QA pair."
    )
    parser.add_argument('--model_dir', type=str, required=True, help='DocLayLLM model directory')
    parser.add_argument(
        '--dataset_dir',
        type=Path,
        default=Path(__file__).resolve().parent / 'dataset' / 'testing_data',
        help='FUNSD testing_data directory (default: %(default)s)',
    )
    parser.add_argument(
        '--output_file',
        type=Path,
        default=Path(__file__).resolve().parent / 'results' / 'funsd_predictions.json',
        help='Prediction JSON path (default: %(default)s)',
    )
    args = parser.parse_args()

    image_dir = args.dataset_dir / 'images'
    ocr_dir = args.dataset_dir / 'ocr_sorted'
    qa_dir = args.dataset_dir / 'qa'
    qa_paths = sorted(qa_dir.glob('*.json'))
    if not qa_paths:
        raise FileNotFoundError(f'No QA JSON files found in: {qa_dir}')

    model, tokenizer, generator_config = load_model(args.model_dir)
    predictions = []
    started_at = time.perf_counter()

    for document_index, qa_path in enumerate(qa_paths, start=1):
        document = qa_path.stem
        image_path = image_dir / f'{document}.png'
        ocr_path = ocr_dir / qa_path.name
        if not image_path.is_file():
            raise FileNotFoundError(f'Missing image for {document}: {image_path}')
        if not ocr_path.is_file():
            raise FileNotFoundError(f'Missing OCR JSON for {document}: {ocr_path}')

        with qa_path.open(encoding='utf-8') as handle:
            qa_pairs = json.load(handle)
        if not isinstance(qa_pairs, list):
            raise ValueError(f'{qa_path}: expected a JSON array of QA pairs')

        print(f'[{document_index}/{len(qa_paths)}] {document}', flush=True)
        for question_index, qa_pair in enumerate(qa_pairs, start=1):
            print(f'    {question_index}/{len(qa_pairs)}', flush=True)
            question = qa_pair['question']
            prediction = predict(
                model,
                tokenizer,
                generator_config,
                str(image_path),
                str(ocr_path),
                question,
            )
            predictions.append(
                {
                    'document': document,
                    'question': question,
                    'ground_truth': qa_pair['answer'],
                    'prediction': prediction,
                }
            )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open('w', encoding='utf-8') as handle:
        json.dump(predictions, handle, ensure_ascii=False, indent=2)
        handle.write('\n')

    elapsed = time.perf_counter() - started_at
    print('Finished.')
    print(f'documents processed: {len(qa_paths)}')
    print(f'questions processed: {len(predictions)}')
    print(f'elapsed time: {elapsed:.2f} seconds')


if __name__ == '__main__':
    main()
