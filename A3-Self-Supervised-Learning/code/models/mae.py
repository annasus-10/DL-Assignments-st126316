import torch
import torch.nn as nn
import numpy as np
import math


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3, embed_dim=192):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)

    def sincos_1d(pos, dim):
        omega = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
        out = pos.reshape(-1, 1) * omega.reshape(1, -1)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    half = embed_dim // 2
    emb = np.concatenate([sincos_1d(grid_h.flatten(), half),
                           sincos_1d(grid_w.flatten(), half)], axis=1)
    return torch.tensor(emb, dtype=torch.float32)


class MAEEncoder(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3,
                 embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0,
                 mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, embed_dim)
        pos_embed = get_2d_sincos_pos_embed(embed_dim, img_size // patch_size)
        self.register_buffer('pos_embed', pos_embed.unsqueeze(0))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def random_masking(self, x):
        N, L, D = x.shape
        n_keep = int(L * (1 - self.mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :n_keep]
        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(N, L, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_visible, mask, ids_restore

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x_vis, mask, ids_restore = self.random_masking(x)
        x_vis = self.norm(self.transformer(x_vis))
        return x_vis, mask, ids_restore


class MAEDecoder(nn.Module):
    def __init__(self, n_patches, patch_size=4, in_ch=3,
                 encoder_dim=192, decoder_dim=128,
                 depth=4, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        patch_pixels = patch_size * patch_size * in_ch
        grid_size = int(math.sqrt(n_patches))
        self.embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        pos_embed = get_2d_sincos_pos_embed(decoder_dim, grid_size)
        self.register_buffer('pos_embed', pos_embed.unsqueeze(0))
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=num_heads,
            dim_feedforward=int(decoder_dim * mlp_ratio),
            dropout=0.0, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_pixels)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x_vis, ids_restore):
        N = x_vis.size(0)
        x = self.embed(x_vis)
        n_masked = ids_restore.size(1) - x.size(1)
        mask_tokens = self.mask_token.expand(N, n_masked, -1)
        x_full = torch.cat([x, mask_tokens], dim=1)
        x_full = torch.gather(
            x_full, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, x_full.size(-1))
        )
        x_full = x_full + self.pos_embed
        x_full = self.norm(self.transformer(x_full))
        return self.pred(x_full)


class MAE(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3,
                 encoder_dim=192, encoder_depth=6, encoder_heads=3,
                 decoder_dim=128, decoder_depth=4, decoder_heads=4,
                 mask_ratio=0.75, norm_pix_loss=True):
        super().__init__()
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.norm_pix_loss = norm_pix_loss
        self.encoder = MAEEncoder(
            img_size, patch_size, in_ch,
            encoder_dim, encoder_depth, encoder_heads,
            mask_ratio=mask_ratio
        )
        n_patches = self.encoder.patch_embed.n_patches
        self.decoder = MAEDecoder(
            n_patches, patch_size, in_ch,
            encoder_dim, decoder_dim, decoder_depth, decoder_heads
        )

    def patchify(self, imgs):
        p = self.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], self.in_ch, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        return x.reshape(imgs.shape[0], h * w, p * p * self.in_ch)

    def forward(self, imgs):
        x_vis, mask, ids_restore = self.encoder(imgs)
        pred = self.decoder(x_vis, ids_restore)
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss, pred, mask