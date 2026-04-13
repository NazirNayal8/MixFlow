from .distribution import KLDivergenceGaussian, KLDivergenceGaussianCfg


LOSS_FUNCTIONS = {
    "kl_divergence_gaussian": KLDivergenceGaussian,
}

LossCfg = KLDivergenceGaussianCfg


def get_loss(cfg: LossCfg):
    return LOSS_FUNCTIONS[cfg.name](cfg)
