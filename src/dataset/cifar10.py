import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10 as TorchCIFAR10
from src.misc.step_tracker import StepTracker
from dataclasses import dataclass
from typing import Literal


@dataclass
class CIFAR10Cfg:
    name: Literal["cifar10"]
    root: str


class CIFAR10(Dataset):
    def __init__(self, cfg: dict, stage: str, step_tracker: StepTracker):
        super().__init__()
        self.root = cfg.root
        self.stage = stage

        if stage in ["train", "test"]:
            train = True
        else:
            train = False

        self.dataset = TorchCIFAR10(
            cfg.root, train=train, download=True
        )

        # normalize from [0, 1] to [-1, 1]

        if stage == "train":
            self.transforms = T.Compose(
                [
                    T.ToTensor(),
                    T.RandomHorizontalFlip(),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            self.transforms = T.Compose(
                [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        x, y = self.dataset[idx]

        x = self.transforms(x)

        return {
            "image": x,
            "label": y,
        }
