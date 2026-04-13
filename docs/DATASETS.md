# Datasets

This release currently documents the three paper experiments kept in [`config/experiment/`](../config/experiment).

## Overview

| Dataset | Configs | How data is obtained | Default root override |
| --- | --- | --- | --- |
| CIFAR-10 | `cifar10` | Downloaded automatically by `torchvision` | `MIXFLOW_CIFAR10_ROOT` |
| FFHQ 64x64 | `ffhq_64x64` | Local image folder expected | `MIXFLOW_FFHQ_ROOT` |
| AFHQv2 64x64 | `afhqv2_64x64` | Local image folder expected | `MIXFLOW_AFHQV2_ROOT` |

## CIFAR-10

The CIFAR-10 loader uses `torchvision.datasets.CIFAR10(..., download=True)`, so the dataset will be downloaded automatically if it is missing.

You can set the root in either of these ways:

```bash
export MIXFLOW_CIFAR10_ROOT=/path/to/datasets
```

or:

```bash
bash scripts/run_train.sh cifar10 dataset.root=/path/to/datasets
```

## FFHQ 64x64

Download the original FFHQ images from the official source:

- https://github.com/NVlabs/ffhq-dataset

Then convert them to the 64x64 archive used by this project with [`scripts/dataset_tool.py`](../scripts/dataset_tool.py):

```bash
python scripts/dataset_tool.py --source=downloads/ffhq/images1024x1024 \
  --dest=datasets/ffhq-64x64.zip --resolution=64x64
```

The `ffhqv2` loader expects a local directory and scans PNG files from subdirectories under `dataset.root`. If you extract the generated archive, point `dataset.root` to the extracted folder.

Expected layout:

```text
/path/to/ffhq-64x64/
├── subset_a/
│   ├── 000001.png
│   ├── 000002.png
│   └── ...
├── subset_b/
│   ├── 000101.png
│   └── ...
└── ...
```

If your dataset is stored in a single folder, place the images inside one subdirectory so the loader can discover them consistently.

Example:

```bash
export MIXFLOW_FFHQ_ROOT=/path/to/ffhq-64x64
bash scripts/run_train.sh ffhq_64x64
```

## AFHQv2 64x64

Download AFHQ from the official source:

- https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq

Then convert it to the 64x64 archive used by this project with [`scripts/dataset_tool.py`](../scripts/dataset_tool.py):

```bash
python scripts/dataset_tool.py --source=downloads/afhqv2 \
  --dest=datasets/afhqv2-64x64.zip --resolution=64x64
```

The `afhqv2` loader expects a local directory with PNG files grouped in subdirectories. If you extract the generated archive, point `dataset.root` to the extracted folder.

Expected layout:

```text
/path/to/afhqv2-64x64/
├── cat/
│   ├── 000001.png
│   └── ...
├── dog/
│   ├── 000001.png
│   └── ...
├── wild/
│   ├── 000001.png
│   └── ...
```

Example:

```bash
export MIXFLOW_AFHQV2_ROOT=/path/to/afhqv2-64x64
bash scripts/run_train.sh afhqv2_64x64
```

## Validation and Test Behavior

- CIFAR-10 test statistics are computed on 50k samples by default.
- FFHQ and AFHQ release experiments use 10k-sample FID evaluation in the paper.
- Real-dataset FID statistics are cached automatically to `./stats/<dataset>_fid_stats.npz`.