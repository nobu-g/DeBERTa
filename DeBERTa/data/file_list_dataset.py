# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: penhe@microsoft.com
# Date: 05/15/2019
#

import json
import random
from pathlib import Path
from typing import Callable

import datasets
import numpy as np
from torch.utils.data import Dataset

from ..data import ExampleInstance
from ..utils import get_logger

logger = get_logger()

__all__ = ["FileListDataset"]

CORPUS_TO_NUM_EXAMPLES: dict[str, int] = {
    "ja_wiki": 2_850_903,  # 2_850_903
    "ja_cc": 1_000_000_000,
    "en_wiki": 11_101_940,
    "en_pile": 0,
    "code_stack": 1_553_108,
}


class FileListDataset(Dataset):
    NUM_EXAMPLES_PER_FILE = 100_000

    def __init__(self, data_dir: Path, glob_pat: str, feature_fn: Callable, shuffle: bool = False):
        self.data_files = sorted(list(data_dir.glob(glob_pat)))
        self.num_data_files = len(self.data_files)
        self.feature_fn: Callable = feature_fn

        self.dataset_size = sum(CORPUS_TO_NUM_EXAMPLES.values())
        logger.info(f"Total estimated corpus examples: {self.dataset_size}")

        self.shuffle: bool = shuffle
        if self.shuffle:
            rng = random.Random(0)
            rng.shuffle(self.data_files)
        self.buff: list[list[str]] = []
        self.data_file_index = 0
        self._load_files()

    def _load_files(self) -> None:
        buff_files = 5
        for _ in range(buff_files):
            self.buff += self._load_file(self.data_files[self.data_file_index % self.num_data_files])
            self.data_file_index += 1

    def _load_file(self, file_path: Path) -> list[list[str]]:
        if file_path.suffix == ".parquet":
            dataset: datasets.Dataset = datasets.Dataset.from_parquet(str(file_path), keep_in_memory=True)
            return dataset["tokens"]
        else:
            assert file_path.suffix == ".jsonl"
            return [json.loads(line)["tokens"] for line in file_path.read_text().splitlines()]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index: int | np.ndarray):
        index = int(index)
        rng = random.Random(index)
        buff_size = len(self.buff)
        if buff_size == 0:
            self._load_files()
            buff_size = len(self.buff)
        example = ExampleInstance(segments=[self.buff.pop(index % buff_size)])
        return self.feature_fn(example, rng, ext_params=None)
