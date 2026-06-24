"""Dataset loaders: MNIST (GAN/DDPM) and CelebA hair-split (CycleGAN)."""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_mnist_loader(data_root="data", batch_size=128, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = torchvision.datasets.MNIST(
        data_root, train=train, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2)


def get_celeba_loaders(data_root="data", batch_size=16, max_per_domain=5000):
    IMG_SIZE = 64
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    torchvision.datasets.CelebA._check_integrity = lambda self: True

    full = torchvision.datasets.CelebA(
        data_root,
        split="train",
        target_type="attr",
        download=False,
        transform=transform,
    )

    BLONDE_ATTR = 9
    dark_idx   = [i for i, (_, attr) in enumerate(full) if attr[BLONDE_ATTR] == 0][:max_per_domain]
    blonde_idx = [i for i, (_, attr) in enumerate(full) if attr[BLONDE_ATTR] == 1][:max_per_domain]

    dark_set   = torch.utils.data.Subset(full, dark_idx)
    blonde_set = torch.utils.data.Subset(full, blonde_idx)

    loader_dark   = DataLoader(dark_set,   batch_size=batch_size, shuffle=True,
                               num_workers=2, drop_last=True)
    loader_blonde = DataLoader(blonde_set, batch_size=batch_size, shuffle=True,
                               num_workers=2, drop_last=True)

    print(f"Domain X (dark hair):   {len(dark_set)} images")
    print(f"Domain Y (blonde hair): {len(blonde_set)} images")
    return loader_dark, loader_blonde