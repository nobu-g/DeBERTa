import logging
from argparse import ArgumentParser
from pathlib import Path

from datasets import Dataset, disable_caching
from tqdm import tqdm
from utils import list_input_files

logger = logging.getLogger(__name__)
disable_caching()


def save_dataset(dataset: Dataset, output_file: Path, overwrite: bool, format: str) -> None:
    if output_file.exists() and not overwrite:
        logger.error(f"{output_file} already exists. Specify --overwrite to overwrite.")
        return
    if format == "jsonl":
        dataset.to_json(output_file, force_ascii=False)
    else:
        assert format == "parquet"
        dataset.to_parquet(output_file)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--input-path",
        "-i",
        type=str,
        nargs="+",
        help="Path(s) to the input data directory or file.",
    )
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the output directory.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=-1,
        help="Number of processes for parallel execution.",
    )
    args = parser.parse_args()

    input_files: list[Path] = sorted(list_input_files(args.input_path, args.input_format))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_file in tqdm(input_files):
        logger.info(f"Loading dataset from {input_file}.")
        if args.input_format == "jsonl":
            dataset = Dataset.from_json(str(input_file), keep_in_memory=True)
        else:
            assert args.input_format == "parquet"
            dataset = Dataset.from_parquet(str(input_file), keep_in_memory=True)
        logger.info(f"Splitting a dataset in {input_file.stem}.")
        num_examples = len(dataset)
        for i in range(0, num_examples, 50_000):
            output_file = output_dir / f"{input_file.stem}_{i}.parquet"
            if output_file.exists() and not args.overwrite:
                logger.error(f"{output_file} already exists. Specify --overwrite to overwrite.")
                continue
            split_dataset = Dataset.from_dict(dataset[i : i + 50_000])
            logger.info(f"Writing the split dataset to {output_file}.")
            save_dataset(split_dataset, output_file, overwrite=args.overwrite, format=args.output_format)
            logger.info(f"Finished writing to {output_file}.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
