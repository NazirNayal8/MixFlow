import torch 
import numpy as np
from pathlib import Path
from torchmetrics.image.fid import FrechetInceptionDistance

class OptimizedFID(FrechetInceptionDistance):
    """
    Frechet Inception Distance that automatically caches / re-uses
    the real-dataset statistics (μ, Σ, N) on disk.

    Parameters
    ----------
    dataset_name : str            # name of the real dataset
    stats_dir    : str or Path     # default: "./stats"
    *args, **kwargs :              # all stock FID arguments
    """
    def __init__(self, dataset_name: str,
                 stats_dir: str | Path = "./stats",
                 cache_key: str | None = None,
                 *args, **kwargs):

        self.dataset_name = dataset_name
        self.stats_dir    = Path(stats_dir)
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        self.cache_key = cache_key or "default"

        # file that will hold npz with keys: mu, sigma, n
        self.stats_file = self.stats_dir / f"{dataset_name}_{self.cache_key}_fid_stats.npz"

        super().__init__(*args, **kwargs)            # build base metric

        # ------------------------------------------------------------
        # 1) If stats already exist ⇒ load & freeze further real updates
        # ------------------------------------------------------------
        if self.stats_file.exists():
            self._load_real_stats()
            self._real_is_frozen = True
            print(f"[FID] Loaded cached real stats for '{dataset_name}' "
                  f"from {self.stats_file}.")
        else:
            self._real_is_frozen = False
            print(f"[FID] No cached stats for '{dataset_name}'. "
                  f"Will compute and save them to {self.stats_file}.")

    # ------------------------------------------------------------
    # Helper: ignore .update(..., real=True) once stats are frozen
    # ------------------------------------------------------------
    @torch.no_grad()
    def update(self, imgs: torch.Tensor, real: bool):
        if real and getattr(self, "_real_is_frozen", False):
            return
        super().update(imgs, real=real)

    # ------------------------------------------------------------
    # Helper: load μ, Σ, N from disk and plug into internal buffers
    # ------------------------------------------------------------
    def _load_real_stats(self):
        data = np.load(self.stats_file)
        mu      = torch.from_numpy(data["mu"]).double()
        sigma   = torch.from_numpy(data["sigma"]).double()
        n       = int(data["n"])

        # TorchMetrics stores   sum(x),  sum(xxᵀ),  #samples
        self.real_features_num_samples = torch.tensor(n)
        self.real_features_sum         = mu * n
        # cov_sum = Σxxᵀ  = (σ + μμᵀ) * (n - 1)
        self.real_features_cov_sum     = (sigma + mu[:, None] @ mu[None, :]) * (n - 1)

    # ------------------------------------------------------------
    # Override compute(): after the normal FID calc, optionally save
    # ------------------------------------------------------------
    def compute(self):
        fid_value = super().compute()          # do the heavy lifting

        # If we just finished computing real stats for the first time,
        # dump them on disk so subsequent runs can reuse them
        if not getattr(self, "_real_is_frozen", False):
            self._save_real_stats()
            self._real_is_frozen = True

        return fid_value

    def _save_real_stats(self):
        n   = int(self.real_features_num_samples)
        mu  = (self.real_features_sum / n).double().cpu().numpy()

        # unbiased covariance: σ = Σ(xxᵀ)/(n-1)  - μμᵀ
        cov_sum = self.real_features_cov_sum.double().cpu().numpy()
        sigma   = (cov_sum / (n - 1) - np.outer(mu, mu))

        np.savez(self.stats_file, mu=mu, sigma=sigma, n=n)
        print(f"[FID] Saved real stats for '{self.dataset_name}' "
              f"to {self.stats_file}.")
