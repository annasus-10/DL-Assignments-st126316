import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 10


class InceptionModule(nn.Module):
    """
    Inception module with 4 parallel branches:
    1x1 conv | 1x1 -> 3x3 conv | 1x1 -> 5x5 conv | 3x3 maxpool -> 1x1 conv
    Outputs are concatenated along the channel dimension.
    """
    def __init__(self, in_channels, out_1x1, out_3x3_reduce, out_3x3,
                 out_5x5_reduce, out_5x5, out_pool_proj):
        super().__init__()

        # Branch 1: 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 1x1 reduce -> 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_3x3_reduce, out_3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Branch 3: 1x1 reduce -> 5x5 conv
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_5x5_reduce, out_5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )

        # Branch 4: 3x3 maxpool -> 1x1 proj
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class SideClassifier(nn.Module):
    """
    Auxiliary classifier attached at two intermediate points in GoogLeNet.
    Used during training only to combat vanishing gradients.
    """
    def __init__(self, in_channels, num_classes=NUM_CLASSES):
        super().__init__()
        self.pool    = nn.AdaptiveAvgPool2d((4, 4))  # fix to always 4x4
        self.conv    = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.relu    = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(128 * 4 * 4, 1024)
        self.dropout = nn.Dropout(p=0.7)
        self.fc2     = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.relu(self.conv(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class GoogLeNet(nn.Module):
    """
    GoogLeNet (Inception v1) with correct backbone and two side classifiers.
    Returns (main_out, aux1_out, aux2_out) during training,
    and just main_out during eval.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # --- Backbone (before first Inception module) ---
        # Matches the original paper: conv -> pool -> conv -> conv -> pool
        self.conv1   = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2   = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # --- Inception modules ---
        self.inception3a = InceptionModule(192,  64,  96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.pool3       = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionModule(480, 192,  96, 208, 16,  48,  64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24,  64,  64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24,  64,  64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32,  64,  64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.pool4       = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        # --- Side classifiers (attached after 4a and 4d) ---
        self.aux1 = SideClassifier(512, num_classes)
        self.aux2 = SideClassifier(528, num_classes)

        # --- Final classifier ---
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout  = nn.Dropout(p=0.4)
        self.flatten  = nn.Flatten()
        self.fc       = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)

        x = self.inception4a(x)
        aux1_out = self.aux1(x) if self.training else None

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2_out = self.aux2(x) if self.training else None

        x = self.inception4e(x)
        x = self.pool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)

        if self.training:
            return x, aux1_out, aux2_out
        return x