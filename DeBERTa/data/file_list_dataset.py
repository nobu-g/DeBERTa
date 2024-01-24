# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: penhe@microsoft.com
# Date: 05/15/2019
#

import random
from pathlib import Path
from typing import Callable

import numpy as np
from datasets import Dataset, concatenate_datasets
from torch.utils.data import Dataset as TorchDataset

from ..data import ExampleInstance
from ..utils import get_logger

logger = get_logger()

__all__ = ["FileListDataset"]

CORPUS_TO_NUM_EXAMPLES: dict[str, int] = {
    "ja_wiki": 2_848_756,
    "ja_cc": 267_019_300,
    "en_wiki": 11_101_940,
    "en_pile": 303_448_037,
    "code_stack": 20_907_722,
}


class FileListDataset(TorchDataset):
    def __init__(self, data_dir: Path, glob_pat: str, feature_fn: Callable, shuffle: bool = False):
        self.data_files = sorted(list(data_dir.glob(glob_pat)))
        self.num_data_files = len(self.data_files)
        self.feature_fn: Callable = feature_fn

        self.dataset_size: int = sum(CORPUS_TO_NUM_EXAMPLES.values())
        logger.info(f"Total corpus examples: {self.dataset_size}")

        self.shuffle: bool = shuffle
        self.rng = random.Random(0)
        if self.shuffle:
            self.rng.shuffle(self.data_files)
        self.data_file_index = 0
        self.buff: Dataset = self._load_files()
        self.offset: int = 0

    def _load_files(self) -> Dataset:
        n_buff_files = 10
        datasets = []
        for _ in range(n_buff_files):
            data_file = self.data_files[self.data_file_index % self.num_data_files]
            datasets.append(self._load_file(data_file))
            logger.info(f"Loaded {data_file}")
            self.data_file_index += 1
        return concatenate_datasets(datasets).shuffle(seed=0)

    def _load_file(self, file_path: Path) -> Dataset:
        assert file_path.suffix == ".parquet"
        return Dataset.from_parquet(str(file_path))

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int | np.ndarray):
        index = int(index)
        buff_size = len(self.buff)
        if index - self.offset >= buff_size:
            self.offset += buff_size
            self.buff = self._load_files()
        example = ExampleInstance(segments=[self.buff[index - self.offset]["tokens"]])
        return self.feature_fn(example, self.rng, ext_params=None)
