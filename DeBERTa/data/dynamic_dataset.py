# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: penhe@microsoft.com
# Date: 05/15/2019
#

import mmap
import random
from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset

from ..utils import get_logger

logger = get_logger()

__all__ = ["DynamicDataset", "SimpleDataset"]


class DynamicDataset(Dataset):
    def __init__(self, corpus, feature_fn, dataset_size=None, shuffle=False, **kwargs):
        self.corpus = corpus
        self.ds_len = len(self.corpus)
        logger.info(f"Total corpus examples: {self.ds_len}")
        self.feature_fn = feature_fn

        if not dataset_size:
            self.dataset_size = self.ds_len
        else:
            self.dataset_size = int(dataset_size)

        self.shuffle = shuffle
        index_buf = mmap.mmap(-1, self.dataset_size * 8)
        shuffle_idx = np.ndarray(shape=(self.dataset_size,), buffer=index_buf, dtype=int)
        shuffle_idx[:] = np.arange(self.dataset_size)[:]
        if self.shuffle:
            # rng = np.random.RandomState(0)
            rng = random.Random(0)
            rng.shuffle(shuffle_idx)
        self.shuffle_idx = shuffle_idx
        self.index_offset = 0
        if "index_offset" in kwargs:
            self.index_offset = kwargs["index_offset"]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple)):
            idx, ext_params = idx
        else:
            ext_params = None
        idx += self.index_offset
        idx = int(idx)
        seed = idx
        rng = random.Random(seed)
        # get seq length
        example_idx = self.shuffle_idx[idx % self.dataset_size] % self.ds_len
        example = self.corpus[example_idx, rng, ext_params]
        return self.feature_fn(example, rng, ext_params=ext_params)


class SimpleDataset(Dataset):
    def __init__(self, examples: list, feature_fn: Callable, dataset_size: Optional[int] = None, **kwargs):
        self.examples: list = examples
        self.feature_fn: Callable = feature_fn
        self.dataset_size: int = dataset_size or len(self.examples)
        logger.info(f"Total corpus examples: {self.dataset_size}")
        # self.shuffle = shuffle

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index: int):
        index = int(index)
        example = self.examples[index]
        rng = random.Random(index)
        return self.feature_fn(example, rng, ext_params=None)
