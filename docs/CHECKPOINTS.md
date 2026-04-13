# Checkpoints

This file is the release index for pretrained MixFlow models.

## Official Releases

| Experiment | Config | Hugging Face Repo | Checkpoint | Download |
| --- | --- | --- | --- | --- |
| CIFAR-10 | `cifar10` | [`nazirnayal98/MixFlow-CIFAR10`](https://huggingface.co/nazirnayal98/MixFlow-CIFAR10) | `mixflow_cifar10.ckpt` | [download](https://huggingface.co/nazirnayal98/MixFlow-CIFAR10/resolve/main/mixflow_cifar10.ckpt) |
| FFHQ 64x64 | `ffhq_64x64` | [`nazirnayal98/MixFlow-FFHQ-64x64`](https://huggingface.co/nazirnayal98/MixFlow-FFHQ-64x64) | `mixflow_ffhqv2_64x64` | [download](https://huggingface.co/nazirnayal98/MixFlow-FFHQ-64x64/resolve/main/mixflow_ffhqv2_64x64) |
| AFHQv2 64x64 | `afhqv2_64x64` | [`nazirnayal98/MixFlow-AFHQ-64x64`](https://huggingface.co/nazirnayal98/MixFlow-AFHQ-64x64) | `mixflow_afhqv2_64x64` | [download](https://huggingface.co/nazirnayal98/MixFlow-AFHQ-64x64/resolve/main/mixflow_afhqv2_64x64) |

## Recommended Commands

### CIFAR-10

```bash
bash scripts/run_inference.sh cifar10 mixflow_cifar10.ckpt outputs/inference/cifar10 \
  test.num_inference_timesteps=9 \
  test.ode_solver=manual_heun
```

Reported paper numbers:

- RK45: `FID = 2.27` at `NFE = 124.7`
- Heun, 5 NFE: `FID = 19.29`
- Heun, 9 NFE: `FID = 8.97`

### FFHQ 64x64

```bash
bash scripts/run_inference.sh ffhq_64x64 mixflow_ffhqv2_64x64 outputs/inference/ffhq \
  test.num_samples=10000 \
  test.num_inference_timesteps=64
```

Reported paper numbers:

- 64 steps: `FID-10k = 4.01`
- 128 steps: `FID-10k = 3.75`

### AFHQv2 64x64

```bash
bash scripts/run_inference.sh afhqv2_64x64 mixflow_afhqv2_64x64 outputs/inference/afhq \
  test.num_samples=10000 \
  test.num_inference_timesteps=64
```

Reported paper numbers:

- 64 steps: `FID-10k = 3.65`
- 128 steps: `FID-10k = 3.33`

## Notes

- All checkpoints correspond to the official paper configs in [`config/experiment/`](../config/experiment).
- You can replace the local checkpoint path in the commands above with the Hugging Face download URL or a locally downloaded copy.
- For FFHQ and AFHQv2, the paper reports FID-10k across multiple sampling budgets; see [README.md](../README.md) for the full comparison tables.
