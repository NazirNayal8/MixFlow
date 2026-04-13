import os
import imageio
import torchvision.transforms as T
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Literal
from src.misc.step_tracker import StepTracker

@dataclass
class FFHQCfg:
    name: Literal["ffhq", "ffhqv2"]
    root: str
    num_val_samples: int


class FFHQv2(Dataset):
    
    def __init__(self, cfg:FFHQCfg, stage: str, step_tracker: StepTracker):
        super().__init__()
        self.root = cfg.root
        self.stage = stage

        self.paths = []
        self.labels = []

        for i, f in enumerate(os.listdir(cfg.root)):
            if not os.path.isdir(os.path.join(cfg.root, f)):
                continue
            for file in os.listdir(os.path.join(cfg.root, f)):
                if file.endswith(".png"):
                    self.paths.append(os.path.join(cfg.root, f, file))
                    self.labels.append(i)

        if stage == "val":
            step = max(1, len(self.paths) // cfg.num_val_samples)
            indices = list(range(0, len(self.paths), step))[: cfg.num_val_samples]
            self.paths = [self.paths[idx] for idx in indices]
            self.labels = [self.labels[idx] for idx in indices]

            self.transforms = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        elif stage == "train":
            self.transforms = T.Compose(
                [
                    T.ToTensor(),
                    T.RandomHorizontalFlip(),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        elif stage == "test":
            self.transforms = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        x = imageio.imread(self.paths[idx])
        y = self.labels[idx]
        
        x = self.transforms(x)

        return {
            "image": x,
            "label": y,
        }
