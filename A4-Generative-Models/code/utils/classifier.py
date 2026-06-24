"""Small MNIST CNN classifier for the GAN mode-collapse check (Exercise 1)."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train_or_load_classifier(device, save_path="saved/mnist_classifier.pt",
                              data_root="data", epochs=5):
    model = MnistCNN().to(device)
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()
        print(f"Loaded MNIST classifier from {save_path}")
        return model

    import torchvision, torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    loader = DataLoader(
        torchvision.datasets.MNIST(data_root, train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])),
        batch_size=256, shuffle=True, num_workers=2
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        correct = total = 0
        for x, y in tqdm(loader, desc=f"Classifier epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += (model(x).argmax(1) == y).sum().item()
            total   += y.size(0)
        print(f"  Epoch {epoch+1} | acc {correct/total*100:.2f}%")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved classifier -> {save_path}")
    model.eval()
    return model