import torchvision
import torchvision.transforms as transforms
import torch

def get_dataloaders(batch_size=64, img_size=224, num_workers=2):
    """
    Returns train, val, test dataloaders for CIFAR-10.
    img_size: 224 for CNN models, 224 for ViT-B/16 (already 224),
              32 for ViTSmall from scratch (pass img_size=32)
    """
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=preprocess)

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [40000, 10000])

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=preprocess)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader