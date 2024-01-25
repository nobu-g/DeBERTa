import logging
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from datasets import Dataset
from tqdm import tqdm
from utils import list_input_files

logger = logging.getLogger(__name__)
disable_caching()


def process_file(input_file: Path, input_format: str, output_dir: Path, output_format: str, overwrite: bool) -> None:
    logger.info(f"Writing the dataset to {output_dir} in {output_format} format.")
    output_file: Path = output_dir.joinpath(f"{input_file.stem}.{output_format}")
    if output_file.exists() and not overwrite:
        logger.error(f"{output_file} already exists. Specify --overwrite to overwrite.")
        return
    if input_format == "jsonl":
        dataset = Dataset.from_json(str(input_file), keep_in_memory=True)
    else:
        assert input_format == "parquet"
        dataset = Dataset.from_parquet(str(input_file), keep_in_memory=True)
    if output_format == "jsonl":
        dataset.to_json(output_file, force_ascii=False)
    else:
        assert output_format == "parquet"
        dataset.to_parquet(output_file)
    logger.info(f"Finished exporting to {output_file}.")


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
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Path to the output directory.",
    )
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
        default=1,
        help="Number of processes for parallel execution.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.num_proc) as executor:
        for input_file in tqdm(list_input_files(args.input_path, args.input_format)):
            executor.submit(process_file, input_file, args.input_format, output_dir, args.output_format, args.overwrite)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
