# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: penhe@microsoft.com
# Date: 05/15/2019
#

import random
from typing import Callable

import datasets
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from ..data import ExampleInstance
from ..utils import get_logger

logger = get_logger()

__all__ = ["FileListDataset"]


class FileListDataset(Dataset):
    NUM_EXAMPLES_PER_FILE = 100_000

    def __init__(self, data_dir: Path, glob_pat: str, feature_fn: Callable, shuffle: bool = False):
        self.data_files = sorted(list(data_dir.glob(glob_pat)))
        self.feature_fn: Callable = feature_fn

        self.dataset_size = (len(self.data_files) - 5) * self.NUM_EXAMPLES_PER_FILE
        logger.info(f"Total estimated corpus examples: {self.dataset_size}")

        self.shuffle: bool = shuffle
        if self.shuffle:
            rng = random.Random(0)
            rng.shuffle(self.data_files)
        self.buff = []
        self.buff_files = 3
        self.data_file_index = 0
        for _ in range(self.buff_files):
            self.buff += self._load_file(self.data_files[self.data_file_index])
            self.data_file_index += 1

    def _load_file(self, file_path: Path) -> list[list[str]]:
        dataset: datasets.Dataset = datasets.Dataset.from_parquet(str(file_path), keep_in_memory=True)
        return dataset["tokens"]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index: int | np.ndarray):
        index = int(index)
        rng = random.Random(index)
        buff_size = len(self.buff)
        if buff_size == 0:
            for _ in range(self.buff_files):
                self.buff += self._load_file(self.data_files[self.data_file_index])
                self.data_file_index += 1
            buff_size = len(self.buff)
        example = ExampleInstance(segments=[self.buff.pop(index % buff_size)])
        return self.feature_fn(example, rng, ext_params=None)
