import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.unet import DoubleConv


class UNetResNet18(nn.Module):
    """U-Net with pretrained ResNet-18 encoder. Decoder restores full input resolution.

    ResNet-18 on 128x128 input:
      stem_conv (stride=2) → 64x64   [s0, skip]
      stem_pool (stride=2) → 32x32
      layer1               → 32x32   [s1, skip]
      layer2 (stride=2)    → 16x16   [s2, skip]
      layer3 (stride=2)    → 8x8     [s3, skip]
      layer4 (stride=2)    → 4x4     [s4, skip]
    Decoder: 4→8→16→32→64→128 (5 upsamples)
    """

    def __init__(self, n_classes=3, pretrained=True):
        super().__init__()

        weights = 'IMAGENET1K_V1' if pretrained else None
        resnet  = models.resnet18(weights=weights)

        self.stem_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.stem_pool = resnet.maxpool

        self.enc1 = resnet.layer1
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4

        self.bottleneck = DoubleConv(512, 1024)

        self.up4  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(512 + 512, 512)

        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(256 + 256, 256)

        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(128 + 128, 128)

        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(64 + 64, 64)

        self.up0  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = DoubleConv(32 + 64, 32)

        self.output = nn.Conv2d(32, n_classes, kernel_size=1)

    def _cat(self, x, skip):
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:])
        return torch.cat([skip, x], dim=1)

    def forward(self, x):
        s0 = self.stem_conv(x)
        sp = self.stem_pool(s0)
        s1 = self.enc1(sp)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        x = self.bottleneck(s4)

        x = self.up4(x); x = self._cat(x, s4); x = self.dec4(x)
        x = self.up3(x); x = self._cat(x, s3); x = self.dec3(x)
        x = self.up2(x); x = self._cat(x, s2); x = self.dec2(x)
        x = self.up1(x); x = self._cat(x, s1); x = self.dec1(x)
        x = self.up0(x); x = self._cat(x, s0); x = self.dec0(x)

        return self.output(x)


class UNetResNet18NoSkip(nn.Module):
    """Ablation: same ResNet-18 encoder, decoder WITHOUT skip connections.
    Each decoder DoubleConv only receives the upsampled feature — no concat.
    Channel sizes are halved accordingly.
    """

    def __init__(self, n_classes=3, pretrained=True):
        super().__init__()

        weights = 'IMAGENET1K_V1' if pretrained else None
        resnet  = models.resnet18(weights=weights)

        self.stem_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.stem_pool = resnet.maxpool

        self.enc1 = resnet.layer1
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4

        self.bottleneck = DoubleConv(512, 1024)

        self.up4  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)

        self.up3  = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2  = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1  = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.up0  = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.dec0 = DoubleConv(32, 32)

        self.output = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        s0 = self.stem_conv(x)
        sp = self.stem_pool(s0)
        s1 = self.enc1(sp)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        x = self.bottleneck(s4)

        x = self.up4(x); x = self.dec4(x)
        x = self.up3(x); x = self.dec3(x)
        x = self.up2(x); x = self.dec2(x)
        x = self.up1(x); x = self.dec1(x)
        x = self.up0(x); x = self.dec0(x)

        return self.output(x)