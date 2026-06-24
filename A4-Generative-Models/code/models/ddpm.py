"""DDPM architectures + noise schedules (Part 3 / Exercise 4)."""

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device).float()
            * (torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm1    = nn.GroupNorm(8, out_ch)
        self.norm2    = nn.GroupNorm(8, out_ch)

    def forward(self, x, t_emb):
        out = self.conv1(x)
        h   = self.norm1(out * torch.sigmoid(out))
        h   = h + self.time_mlp(t_emb)[:, :, None, None]
        out = self.conv2(h)
        h   = self.norm2(out * torch.sigmoid(out))
        return h + self.residual(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, time_dim=256):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.enc1 = ResBlock(in_ch,     base_ch,   time_dim)
        self.enc2 = ResBlock(base_ch,   base_ch*2, time_dim)
        self.down = nn.MaxPool2d(2)
        self.bot  = ResBlock(base_ch*2, base_ch*4, time_dim)
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = ResBlock(base_ch*4 + base_ch*2, base_ch*2, time_dim)
        self.dec1 = ResBlock(base_ch*2 + base_ch,   base_ch,   time_dim)
        self.out  = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down(e1), t_emb)
        b  = self.bot(self.down(e2), t_emb)
        d2 = self.dec2(torch.cat([self.up(b), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1), t_emb)
        return self.out(d1)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    t = torch.linspace(0, timesteps, timesteps + 1)
    alphas_bar = torch.cos(((t / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def get_schedule(name, timesteps):
    if name == "cosine":
        return cosine_beta_schedule(timesteps)
    return linear_beta_schedule(timesteps)