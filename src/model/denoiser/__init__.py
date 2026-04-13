from .songunet import SongUNetCfg, SongUNet
from typing import Union


DENOISERS = {
    "songunet": SongUNet,
}

DenoiserCfg = Union[
    SongUNetCfg,
]


def get_denoiser(cfg: DenoiserCfg):
    return DENOISERS[cfg.name](cfg)
