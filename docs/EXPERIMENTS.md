# Official Experiments

This cleaned release currently focuses on the three main experiments retained in [`config/experiment/`](../config/experiment).

## Summary

| Config | Dataset | Resolution | Max steps | KL weight (`beta`) | Notes |
| --- | --- | --- | --- | --- | --- |
| `cifar10` | CIFAR-10 | 32x32 | 500k | `1e-5` | Paper comparison against Rectified Flow, Fast-ODE, QAC, and other baselines |
| `ffhq_64x64` | FFHQ | 64x64 | 800k | `5e-5` | Paper high-resolution experiment |
| `afhqv2_64x64` | AFHQv2 | 64x64 | 800k | `5e-5` | Paper high-resolution experiment |

All three experiments use:

- flow matching as the scheduler
- a learnable conditional source distribution
- mixed source training through `train.activate_w_mode=true`
- interpolation between unconditional and conditional source distributions through `train.interpolate_distrubutions=true`

## Paper Numbers

### CIFAR-10

Reported in the paper:

- RK45 full simulation: `FID = 2.27` at `NFE = 124.7`
- Heun, 5 NFE: `FID = 19.29`
- Heun, 9 NFE: `FID = 8.97`

### FFHQ 64x64

Reported FID-10k:

- 4 steps: `33.72`
- 10 steps: `12.23`
- 64 steps: `4.01`
- 128 steps: `3.75`

### AFHQv2 64x64

Reported FID-10k:

- 4 steps: `19.72`
- 10 steps: `7.95`
- 64 steps: `3.65`
- 128 steps: `3.33`

## Training Commands

```bash
bash scripts/run_train.sh cifar10
bash scripts/run_train.sh ffhq_64x64 dataset.root=/path/to/ffhq-64x64
bash scripts/run_train.sh afhqv2_64x64 dataset.root=/path/to/afhqv2-64x64
```

## Inference Commands

### CIFAR-10 low-NFE evaluation

```bash
bash scripts/run_inference.sh cifar10 /path/to/model.ckpt outputs/inference/cifar10 \
  test.num_inference_timesteps=5 \
  test.ode_solver=manual_heun
```

```bash
bash scripts/run_inference.sh cifar10 /path/to/model.ckpt outputs/inference/cifar10 \
  test.num_inference_timesteps=9 \
  test.ode_solver=manual_heun
```

### FFHQ 64x64

```bash
bash scripts/run_inference.sh ffhq_64x64 /path/to/model.ckpt outputs/inference/ffhq \
  test.num_samples=10000 \
  test.num_inference_timesteps=64
```

### AFHQv2 64x64

```bash
bash scripts/run_inference.sh afhqv2_64x64 /path/to/model.ckpt outputs/inference/afhq \
  test.num_samples=10000 \
  test.num_inference_timesteps=64
```

## What Still Needs To Be Filled In

- Any paper-specific evaluation overrides you want to freeze as official release commands beyond the current examples.
- Optional additional experiments if you decide to expand beyond the current three-config release.
