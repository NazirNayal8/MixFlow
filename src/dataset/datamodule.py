import random
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from src.misc.step_tracker import StepTracker
from torch.utils.data import DataLoader, Subset
from torch import Generator
from typing import Optional
from . import get_dataset, DatasetCfg
from dataclasses import dataclass


@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    seed: Optional[int]
    persistent_workers: Optional[bool]

@dataclass
class DataLoaderCfg:
    train: DataLoaderStageCfg
    val: DataLoaderStageCfg
    test: DataLoaderStageCfg


def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


class DataModule(LightningDataModule):

    def __init__(
        self,
        dataset_cfg: DatasetCfg,
        data_loader_cfg: DataLoaderCfg,
        step_tracker: StepTracker,
        global_rank: int,
        test_num_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.data_loader_cfg = data_loader_cfg
        self.step_tracker = step_tracker
        self.global_rank = global_rank
        self.test_num_samples = test_num_samples

    def get_generator(self, loader_cfg: DataLoaderStageCfg) -> Optional[Generator]:
        if loader_cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(loader_cfg.seed + self.global_rank)
        return generator

    def get_persistent(self, loader_cfg: DataLoaderStageCfg) -> Optional[bool]:
        return None if loader_cfg.num_workers == 0 else loader_cfg.persistent_workers
    
    def train_dataloader(self):
        dataset = get_dataset(self.dataset_cfg, "train", self.step_tracker)
        return DataLoader(
            dataset,
            batch_size=self.data_loader_cfg.train.batch_size,
            num_workers=self.data_loader_cfg.train.num_workers,
            generator=self.get_generator(self.data_loader_cfg.train),
            shuffle=True,
            persistent_workers=self.get_persistent(self.data_loader_cfg.train),
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        dataset = get_dataset(self.dataset_cfg, "val", self.step_tracker)
        return DataLoader(
            dataset,
            batch_size=self.data_loader_cfg.val.batch_size,
            num_workers=self.data_loader_cfg.val.num_workers,
            generator=self.get_generator(self.data_loader_cfg.val),
            shuffle=False,
            persistent_workers=self.get_persistent(self.data_loader_cfg.val),
            worker_init_fn=worker_init_fn,
        )
    
    def test_dataloader(self):
        dataset = get_dataset(self.dataset_cfg, "test", self.step_tracker)
        if self.test_num_samples is not None:
            dataset = Subset(dataset, range(min(len(dataset), self.test_num_samples)))
        return DataLoader(
            dataset,
            batch_size=self.data_loader_cfg.test.batch_size,
            num_workers=self.data_loader_cfg.test.num_workers,
            generator=self.get_generator(self.data_loader_cfg.test),
            shuffle=False,
            persistent_workers=self.get_persistent(self.data_loader_cfg.test),
            worker_init_fn=worker_init_fn,
        )
