#!/usr/bin/env python3
"""
A4: Generative Models — unified entrypoint.

Examples
--------
# Train Vanilla GAN on MNIST
python3 run.py --model gan --dataset mnist --epochs 20 --train

# Mode-collapse check (Ex 1a): classify 1000 generated digits
python3 run.py --model gan --weights saved/gan_mnist.pt --evaluate

# Induce mode collapse (Ex 1b): 3x discriminator lr
python3 run.py --model gan --dataset mnist --epochs 20 --d-lr 6e-4 --train --tag collapse

# Train CycleGAN on CelebA
python3 run.py --model cyclegan --dataset celeba --epochs 20 --train

# Cycle-consistency ablation (Ex 2): disable cycle loss
python3 run.py --model cyclegan --dataset celeba --epochs 10 --lambda-cyc 0 --train --tag nocyc

# Test CycleGAN with your own face (Ex 3)
python3 run.py --model cyclegan --weights saved/cyclegan_celeba.pt --test-image my_face.jpg

# Train DDPM on MNIST (linear, then cosine for Ex 4)
python3 run.py --model ddpm --dataset mnist --epochs 20 --train
python3 run.py --model ddpm --dataset mnist --epochs 20 --schedule cosine --train

# Generate DDPM samples
python3 run.py --model ddpm --weights saved/ddpm_mnist.pt --generate --n 64
"""
import argparse
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="A4 Generative Models — GAN / CycleGAN / DDPM")
    p.add_argument("--model", required=True, choices=["gan", "cyclegan", "ddpm"])
    p.add_argument("--dataset", choices=["mnist", "celeba"])

    # actions
    p.add_argument("--train", action="store_true", help="run training")
    p.add_argument("--evaluate", action="store_true", help="run evaluation / metrics")
    p.add_argument("--generate", action="store_true", help="generate samples from weights")

    # io
    p.add_argument("--weights", type=str, default=None, help="checkpoint to load")
    p.add_argument("--test-image", type=str, default=None, help="path to a face image (cyclegan)")
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--save-dir", type=str, default="saved")
    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--tag", type=str, default="", help="suffix for output/checkpoint names")

    # training knobs
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=None, help="override model default")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=64, help="number of samples to generate")

    # exercise-specific knobs
    p.add_argument("--d-lr", type=float, default=2e-4, help="discriminator lr (Ex 1b: 6e-4)")
    p.add_argument("--lambda-cyc", type=float, default=10.0, help="cycle weight (Ex 2: 0)")
    p.add_argument("--schedule", choices=["linear", "cosine"], default="linear",
                   help="DDPM noise schedule (Ex 4)")
    return p


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] model={args.model} device={device} tag='{args.tag}'")

    if not (args.train or args.evaluate or args.generate or args.test_image):
        raise SystemExit("Nothing to do: pass one of --train / --evaluate / --generate / --test-image")

    if args.model == "gan":
        from utils.trainers import train_gan, evaluate_gan, generate_gan
        if args.train:
            train_gan(args, device)
        if args.evaluate:
            evaluate_gan(args, device)
        if args.generate:
            generate_gan(args, device)

    elif args.model == "cyclegan":
        from utils.trainers import train_cyclegan, test_cyclegan_face
        if args.train:
            train_cyclegan(args, device)
        if args.test_image:
            test_cyclegan_face(args, device)

    elif args.model == "ddpm":
        from utils.trainers import train_ddpm, generate_ddpm
        if args.train:
            train_ddpm(args, device)
        if args.generate:
            generate_ddpm(args, device)


if __name__ == "__main__":
    main()
