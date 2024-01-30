import argparse
import logging
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import sentencepiece as spm
from datasets import Dataset, disable_caching
from tqdm import tqdm
from utils import list_input_files

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


def process_dataset(
    input_dataset: Dataset,
    output_file: Path,
    overwrite: bool,
    output_format: str,
    max_seq_length: int,
    num_proc: int,
) -> None:
    logger.info("Tokenizing the dataset.")
    columns = list(input_dataset[0].keys())
    if "pile_set_name" in input_dataset[0]["meta"]["meta"]:
        filter_fn = lambda x: x["meta"]["meta"]["pile_set_name"] != "Books3"  # noqa: E731
    elif "pile_set_name" in input_dataset[0]["meta"]:
        filter_fn = lambda x: x["meta"]["pile_set_name"] != "Books3"  # noqa: E731
    else:
        filter_fn = lambda x: True  # noqa: E731
    tokenized_dataset = (
        input_dataset.filter(filter_fn, num_proc=num_proc, keep_in_memory=True)
        .map(
            tokenize_examples,
            batched=True,
            batch_size=1024,
            keep_in_memory=True,
            num_proc=num_proc,
        )
        .map(remove_columns=list(set(columns) - {"tokens", "token_ids"}))
    )
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
        batch_size=1024,
        num_proc=num_proc,
        desc=f"Grouping texts in chunks of {max_seq_length}",
    )

    logger.info(f"Writing the tokenized data to {output_file}.")
    save_dataset(grouped_dataset, output_file, overwrite=overwrite, format=output_format)
    logger.info(f"Finished writing the tokenized to {output_file}.")


def process_file(
    input_file: Path, input_format: str, output_dir: Path, output_format: str, overwrite: bool, max_seq_length: int
) -> None:
    logger.info("Loading the dataset")
    output_file: Path = output_dir / f"{input_file.stem}.{output_format}"
    if output_file.exists() and not overwrite:
        logger.error(f"{output_file} already exists. Specify --overwrite to overwrite.")
        return
    logger.info(f"Loading dataset from {input_file}.")
    if input_format == "jsonl":
        dataset = Dataset.from_json(str(input_file), keep_in_memory=True)
    else:
        assert input_format == "parquet"
        dataset = Dataset.from_parquet(str(input_file), keep_in_memory=True)
    process_dataset(
        dataset,
        output_file,
        overwrite,
        output_format,
        max_seq_length=max_seq_length - 2,
        num_proc=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", "-i", type=str, nargs="+", help="Path(s) to the input data directory or file.")
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

    input_files: list[Path] = sorted(list_input_files(args.input_path, args.input_format))

    with ProcessPoolExecutor(max_workers=args.num_proc) as executor:
        for input_file in tqdm(input_files):
            executor.submit(
                process_file,
                input_file,
                args.input_format,
                output_dir,
                args.output_format,
                args.overwrite,
                max_seq_length=args.max_seq_length,
            )

    end_time = time.time()
    logger.info(f"Finished tokenizing the dataset. Elapsed time: {end_time - start_time} [sec]")


if __name__ == "__main__":
    main()
