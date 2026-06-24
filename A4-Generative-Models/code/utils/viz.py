"""Plotting helpers for A4."""

import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision


def save_grid(tensor, path, nrow=8, title=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    if title:
        plt.title(title, fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_losses(g_losses, d_losses, path, title="Training Losses"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(g_losses, label="Generator", color="steelblue")
    plt.plot(d_losses, label="Discriminator", color="coral")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(title); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_digit_histogram(counts, path, title="Generated Digit Distribution"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    bars = plt.bar(range(10), counts, color="steelblue", edgecolor="black")
    plt.axhline(y=100, color="red", linestyle="--", label="Uniform (100)")
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(count), ha="center", va="bottom", fontsize=9)
    plt.xticks(range(10)); plt.xlabel("Digit"); plt.ylabel("Count (out of 1000)")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_alpha_bar(ab_linear, ab_cosine, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    t = np.arange(len(ab_linear))
    axes[0].plot(t, ab_linear.cpu().numpy(), color="steelblue")
    axes[0].set_title("Linear schedule — ᾱ_t")
    axes[0].set_xlabel("Timestep t"); axes[0].set_ylabel("ᾱ_t"); axes[0].grid(True)
    axes[1].plot(t, ab_cosine.cpu().numpy(), color="coral")
    axes[1].set_title("Cosine schedule — ᾱ_t")
    axes[1].set_xlabel("Timestep t"); axes[1].grid(True)
    plt.suptitle("Noise Schedule Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")