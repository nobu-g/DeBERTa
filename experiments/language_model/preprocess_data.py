import argparse
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Literal

import sentencepiece as spm
from datasets import Dataset, disable_caching
from tqdm import tqdm

logger = logging.getLogger(__name__)
disable_caching()

sentence_piece_processor: spm.SentencePieceProcessor


def tokenize_examples(examples: dict[str, Any]) -> dict[str, Any]:
    token_ids: list[list[int]] = sentence_piece_processor.encode_as_ids(examples["text"])
    return {
        "tokens": [sentence_piece_processor.id_to_piece(ids) for ids in token_ids],
        "token_ids": token_ids,
    }


def save_dataset(dataset: Dataset, output_file: Path, overwrite: bool, format: str) -> None:
    if output_file.exists() and not overwrite:
        logger.error(f"{output_file} already exists. Specify --overwrite to overwrite.")
        return
    if format == "jsonl":
        dataset.to_json(output_file, force_ascii=False)
    else:
        assert format == "parquet"
        dataset.to_parquet(output_file)


def list_input_files(input_paths: list[str], input_format: Literal["parquet", "jsonl"] = "parquet") -> Iterator[Path]:
    for path_str in input_paths:
        path = Path(path_str)
        if path.exists() is False:
            logger.warning(f"{path} not found and skipped")
            continue
        yield from path.glob(f"*.{input_format}") if path.is_dir() else [path]


def process_file(
    input_dataset: Dataset,
    output_file: Path,
    overwrite: bool,
    output_format: str,
    max_seq_length: int,
    num_proc: int,
) -> None:
    logger.info("Tokenizing the dataset.")
    columns = list(input_dataset[0].keys())
    tokenized_dataset = input_dataset.map(
        tokenize_examples,
        batched=True,
        batch_size=128,
        keep_in_memory=True,
        num_proc=num_proc,
    ).map(remove_columns=list(set(columns) - {"tokens", "token_ids"}))
    logger.info("Finished tokenizing the dataset.")

    def group_texts(example: dict[str, list[list[int | str]]]) -> dict[str, list[list[int | str]]]:
        total_lengths: list[int] = [len(ex) for ex in example["tokens"]]
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        result: dict[str, list[list[int | str]]] = defaultdict(list)
        for key, vals in example.items():
            for val, total_length in zip(vals, total_lengths):
                if total_length >= max_seq_length:
                    total_length = (total_length // max_seq_length) * max_seq_length
                result[key] += [val[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        return result

    grouped_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc=f"Grouping texts in chunks of {max_seq_length}",
    )

    logger.info(f"Writing the tokenized data to {output_file}.")
    save_dataset(grouped_dataset.shuffle(seed=42), output_file, overwrite=overwrite, format=output_format)
    logger.info(f"Finished writing the tokenized to {output_file}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", required=True, help="The input data path")
    parser.add_argument(
        "--input-format",
        type=str,
        default="parquet",
        choices=["jsonl", "parquet"],
        help="Input format.",
    )
    parser.add_argument("-o", "--output-dir", default=None, help="The output data path")
    parser.add_argument(
        "--output-format",
        type=str,
        default="jsonl",
        choices=["jsonl", "parquet"],
        help="Output format.",
    )
    parser.add_argument("--sentencepiece-model", "--spm", type=str, required=True, help="The sentencepiece model path")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length of inputs")
    parser.add_argument(
        "--num-proc",
        type=int,
        default=-1,
        help="Number of processes for parallel execution.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the output directory.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    logger.info("Initialize the tokenizer.")
    global sentence_piece_processor
    sentence_piece_processor = spm.SentencePieceProcessor(args.sentencepiece_model)

    input_files: list[Path] = sorted(list_input_files([args.input_dir], args.input_format))
    if not input_files:
        return

    logger.info("Loading the dataset")
    for input_file in tqdm(input_files):
        logger.info(f"Loading dataset from {input_file}.")
        if args.input_format == "jsonl":
            dataset = Dataset.from_json(str(input_file), keep_in_memory=True)
        else:
            assert args.input_format == "parquet"
            dataset = Dataset.from_parquet(str(input_file), keep_in_memory=True)
        output_file: Path = output_dir / f"{input_file.stem}.{args.output_format}"
        process_file(
            dataset,
            output_file,
            args.overwrite,
            args.output_format,
            max_seq_length=args.max_seq_length - 2,
            num_proc=os.cpu_count() if args.num_proc == -1 else args.num_proc,
        )

    end_time = time.time()
    logger.info(f"Finished tokenizing the dataset. Elapsed time: {end_time - start_time} [sec]")


if __name__ == "__main__":
    main()
