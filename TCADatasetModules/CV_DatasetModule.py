from typing import Optional

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from pathlib import Path

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader
import torchvision
import platform

sys = platform.system()
if sys == "Windows":
    root = "E:\\kaggle\\competitions"
else:
    root = "/home/kaggle/competitions"
class CVDataModule(pl.LightningDataModule):
    def __init__(self, loader_cfg: DictConfig, dataset_name: str, **kwargs):
        super(CVDataModule, self).__init__(**kwargs)
        self.loader_cfg = loader_cfg
        self.dataset_name = dataset_name

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_set = Imageset(self.dataset_name, "train")
            self.val_set = Imageset(self.dataset_name, "val")
        if stage == "test":
            self.test_set = Imageset(self.dataset_name, "test")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set, shuffle=True, **self.loader_cfg)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_set, shuffle=False, **self.loader_cfg)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, shuffle=False, **self.loader_cfg)


class Imageset(Dataset):
    def __init__(self, dataset_name: str, mode: str, **kwargs):
        super(Imageset, self).__init__(**kwargs)
        self.dataset_name = dataset_name
        self.data = self.dataperpared(mode)

    def dataperpared(self, mode1: str):
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        transform1 = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(224),
                                                     torchvision.transforms.ToTensor()])
        if mode1 == "train":
            mode = True
        else:
            mode = False
        if self.dataset_name == "cifar10":
            return torchvision.datasets.CIFAR10(root=root, train=mode,
                                                download=True, transform=transform)
        if self.dataset_name == "fashion":
            return torchvision.datasets.FashionMNIST(root=root, train=mode,
                                                     download=True, transform=transform)
        if self.dataset_name == "imagenet":
            return torchvision.datasets.ImageFolder(root="/home/kaggle/competitions/ImageNet2012/"+mode1,
                                                    transform=transform1)
        if self.dataset_name == "mnist":
            return torchvision.datasets.MNIST(root=root, train=mode,
                                                     download=True, transform=transform)
        if self.dataset_name == "cifar100":
            return torchvision.datasets.CIFAR100(root=root, train=mode,
                                                download=True, transform=transform)
        if self.dataset_name == "flowers102":
            return torchvision.datasets.Flowers102(root=root, split=mode1,
                                                download=True, transform=transform1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        return {
            "image": sample[0],
            "label": sample[1]
        }


def plt(sample):
    import matplotlib.pyplot as plt
    plt.imshow(sample["image"].squeeze(0).permute(1, 2, 0).numpy())
    plt.show()




