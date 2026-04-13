import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Literal


@dataclass
class KLDivergenceGaussianCfg:
    name: Literal["kl_divergence_gaussian"]
    weight: float


class KLDivergenceGaussian(nn.Module):
    """
    Computes the KL divergence between the standard Gaussian and the
    predicted Gaussian parameterized by the noise predictor.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x, noise, denoiser_pred, t, extra_data):
        if "mean" not in extra_data:
            raise ValueError("KLDivergenceGaussian loss requires mean in extra_data")
        if "log_var" not in extra_data:
            raise ValueError("KLDivergenceGaussian loss requires log_var in extra_data")

        mean = extra_data.mean
        log_var = extra_data.log_var
        var = torch.exp(log_var)

        kl = 0.5 * torch.sum(
            torch.pow(mean, 2) + var - 1.0 - log_var, dim=[1, 2, 3]
        )
        return kl.mean()
