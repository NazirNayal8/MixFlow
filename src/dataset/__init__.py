from src.misc.step_tracker import StepTracker
from .cifar10 import CIFAR10, CIFAR10Cfg
from .ffhq import FFHQv2, FFHQCfg
from .afhq import AFHQv2, AFHQv2Cfg
from typing import Optional, Union


DATASETS = {
    "cifar10": CIFAR10,
    "ffhqv2": FFHQv2,
    "afhqv2": AFHQv2,
}

DatasetCfg = Union[
    CIFAR10Cfg,
    FFHQCfg,
    AFHQv2Cfg,
]


def get_dataset(cfg: DatasetCfg, stage: str, step_tracker: Optional[StepTracker]):
    return DATASETS[cfg.name](cfg, stage, step_tracker)
