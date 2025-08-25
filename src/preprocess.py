import os
from typing import Optional

import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
os.makedirs(DATA_DIR, exist_ok=True)


class SyntheticPatternsDataset(Dataset):
    def __init__(self, n: int = 5000, image_size: int = 32, pattern: str = 'mix'):
        super().__init__()
        self.n = n
        self.S = image_size
        self.pattern = pattern

    def __len__(self):
        return self.n

    def _stripes(self):
        x = torch.zeros(3, self.S, self.S)
        x[:, ::2, :] = 1.0
        return x

    def _checker(self):
        x = torch.zeros(3, self.S, self.S)
        for i in range(self.S):
            for j in range(self.S):
                if (i // 2 + j // 2) % 2 == 0:
                    x[:, i, j] = 1.0
        return x

    def _disk(self):
        x = torch.zeros(3, self.S, self.S)
        cx = cy = self.S // 2
        r = self.S // 3
        yy, xx = torch.meshgrid(torch.arange(self.S), torch.arange(self.S), indexing='ij')
        mask = (xx - cx)**2 + (yy - cy)**2 <= r*r
        x[:, mask] = 1.0
        return x

    def _grad(self):
        x = torch.linspace(0, 1, self.S).view(1, 1, -1).repeat(1, self.S, 1)
        x = x.expand(3, -1, -1)
        return x

    def _noise(self):
        return torch.rand(3, self.S, self.S)

    def __getitem__(self, idx):
        import random
        if self.pattern == 'mix':
            pat = random.choice(['stripes', 'checker', 'disk', 'grad', 'noise'])
        else:
            pat = self.pattern
        if pat == 'stripes': x = self._stripes()
        elif pat == 'checker': x = self._checker()
        elif pat == 'disk': x = self._disk()
        elif pat == 'grad': x = self._grad()
        else: x = self._noise()
        x = x * 2 - 1
        return x, 0


def make_dataset(name: str, root: Optional[str], image_size: int) -> Dataset:
    name = name.lower()
    if name == 'synthetic':
        return SyntheticPatternsDataset(n=5000, image_size=image_size, pattern='mix')
    tx = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(image_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    if name == 'cifar10':
        return tv.datasets.CIFAR10(root or DATA_DIR, download=True, transform=tx)
    if name in ['imagenet64', 'imagenet-64', 'lsun', 'lsun_bedrooms', 'celebahq', 'celebahq-512']:
        assert root is not None and os.path.exists(root), 'Provide --data_root pointing to an ImageFolder.'
        return tv.datasets.ImageFolder(root=root, transform=tx)
    raise ValueError(f'Unknown dataset {name}')
