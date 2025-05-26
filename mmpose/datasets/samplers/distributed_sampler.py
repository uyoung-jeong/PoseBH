# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.utils.data import DistributedSampler as _DistributedSampler

import math
from typing import TypeVar, Optional, Iterator
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist
T_co = TypeVar('T_co', covariant=True)

import random

class DistributedSampler(_DistributedSampler):
    """DistributedSampler inheriting from
    `torch.utils.data.DistributedSampler`.

    In pytorch of lower versions, there is no `shuffle` argument. This child
    class will port one to DistributedSampler.
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        """Deterministically shuffle based on epoch."""
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

# https://github.com/pytorch/pytorch/blob/bc2c6edaf163b1a1330e37a6e34caf8c553e4755/torch/utils/data/distributed.py
class MDDistributedSampler(Sampler[T_co]):
    r"""Multi-dataset DistributedSampler.
    Sample data for each dataset with given ratios

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
        ratios (list, optional): sampling ratios for each dataset.
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False,
                 ratios: Optional[list] = None) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        # This implementation only works for ConcatDataset
        self.n_dataset = len(dataset.datasets)
        if ratios is None:
            ratios = [1 for _ in range(self.n_dataset)]
        self.ratios = ratios
        self.num_samples_frac_list = [int(len(dset) * ratio) for dset,ratio in zip(dataset.datasets, ratios)]
        self.cumulative_sizes = dataset.cumulative_sizes
        self.len_dataset_frac = sum(self.num_samples_frac_list)

        if self.drop_last and self.len_dataset_frac % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (self.len_dataset_frac - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self.len_dataset_frac / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

        if rank == 0:
            print(f"Use {self.total_size} samples out of {self.cumulative_sizes[-1]}. shuffle: {shuffle}")

    def fractional_sampling(self, generator=None):
        indices = []
        start_i = 0
        for di in range(self.n_dataset):
            num_frac_di = self.num_samples_frac_list[di]
            end_i = self.cumulative_sizes[di]
            ratio = self.ratios[di]
            indices_di = list(range(start_i,end_i))
            if ratio < 1:
                random.shuffle(indices_di)
                indices_di = indices_di[:num_frac_di]
            indices += indices_di
            start_i = end_i

        if self.shuffle:
            random.shuffle(indices)
        return indices

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            random.seed(self.seed + self.epoch)
        indices = self.fractional_sampling()
        
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas] # [start:end:jump]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
