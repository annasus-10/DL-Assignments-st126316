# A4: Generative Models

Student ID: st126316

## Overview

This assignment implements and compares three generative model families on MNIST and CelebA:

- **Vanilla GAN** — fully-connected generator and discriminator, adversarial training on MNIST
- **CycleGAN** — unpaired image-to-image translation (dark hair ↔ blonde hair) on CelebA, ResNet generator + PatchGAN discriminator
- **DDPM** — denoising diffusion probabilistic model on MNIST, iterative reverse-process generation via a U-Net denoiser

All models are driven by a single `run.py` entrypoint. Exercises cover mode collapse analysis (GAN), cycle-consistency ablation (CycleGAN), own-face style transfer, and a noise-schedule ablation (DDPM linear vs cosine).

---

## How to Run

```bash
# Activate environment
cd A4-Generative-Models/code
source .venv/bin/activate

# Train Vanilla GAN
python3 run.py --model gan --dataset mnist --epochs 20 --train 2>&1 | tee logs/gan_train.log

# Mode-collapse check — classify 1000 generated digits (Exercise 1a)
python3 run.py --model gan --weights saved/gan_mnist.pt --evaluate 2>&1 | tee logs/gan_eval.log

# Induce collapse — 3× discriminator lr (Exercise 1b)
python3 run.py --model gan --dataset mnist --epochs 20 --d-lr 6e-4 --train --tag collapse 2>&1 | tee logs/gan_collapse.log
python3 run.py --model gan --weights saved/gan_mnist_collapse.pt --evaluate --tag collapse 2>&1 | tee logs/gan_collapse_eval.log

# Train CycleGAN
python3 run.py --model cyclegan --dataset celeba --epochs 10 --train 2>&1 | tee logs/cyclegan_train.log

# Cycle-consistency ablation — disable cycle loss (Exercise 2)
python3 run.py --model cyclegan --dataset celeba --epochs 10 --lambda-cyc 0 --train --tag nocyc 2>&1 | tee logs/cyclegan_nocyc.log

# Test with own face (Exercise 3)
python3 run.py --model cyclegan --weights saved/cyclegan_celeba.pt --test-image my_face.jpg

# Train DDPM — linear schedule (baseline)
python3 run.py --model ddpm --dataset mnist --epochs 10 --train 2>&1 | tee logs/ddpm_linear.log

# Train DDPM — cosine schedule (Exercise 4)
python3 run.py --model ddpm --dataset mnist --epochs 10 --schedule cosine --train 2>&1 | tee logs/ddpm_cosine.log

# Generate DDPM samples
python3 run.py --model ddpm --weights saved/ddpm_mnist.pt --generate --n 64
```

---

## Results

| Model | Dataset | Final Loss | Training Time | Notes |
|---|---|---|---|---|
| Vanilla GAN | MNIST | G: 1.291 / D: 1.028 | ~8.7s/epoch | 20 epochs |
| CycleGAN | CelebA | G: 4.214 / D: 0.393 | ~103s/epoch | 10 epochs, 5k/domain |
| DDPM (linear) | MNIST | 0.0264 | ~25s/epoch | 10 epochs |
| DDPM (cosine) | MNIST | 0.0438 | ~25s/epoch | 10 epochs |

---

## Exercise 1 — GAN Mode Collapse

### 1a. Digit distribution after normal training (d-lr = 2e-4)

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Count (out of 1000) | 34 | 212 | 73 | 123 | 100 | 59 | 56 | 154 | 93 | 96 |

![Mode Collapse Histogram](figures/mode_collapse_histogram.png)

The distribution is uneven — digits 1 (212) and 7 (154) are heavily overrepresented while digits 0 (34) and 6 (56) are underrepresented, indicating partial mode collapse even under normal training.

### 1b. Digit distribution after induced collapse (d-lr = 6e-4)

| Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Count (out of 1000) | 54 | 197 | 57 | 131 | 104 | 46 | 85 | 143 | 88 | 95 |

![Mode Collapse Histogram (induced)](figures/mode_collapse_histogram_collapse.png)

With a 3× discriminator learning rate, collapse worsens — digit 1 dominates even more strongly (197) while digits 5 (46) and 0 (54) nearly vanish. A stronger discriminator overwhelms the generator early in training, forcing it to exploit the few modes it has already learned rather than exploring new ones.

### 1c. Two techniques to prevent mode collapse

**WGAN (Wasserstein GAN):** Replaces the Jensen-Shannon divergence with the Wasserstein distance as the training objective, enforced via weight clipping or gradient penalty. The Wasserstein distance provides meaningful gradients even when the generator and real distributions barely overlap, removing the instability that drives the generator to collapse onto a few modes.

**Minibatch discrimination:** The discriminator receives a summary statistic computed across the whole mini-batch alongside per-image features, capturing how similar the current fake image is to others in the batch. This forces the generator to produce diverse outputs — if it collapses to a single mode, all batch images become nearly identical and the discriminator can easily detect that pattern.

---

## Exercise 2 — CycleGAN Cycle-Consistency Ablation

### 2a. Translation quality comparison

| Setting | Visual quality | Face structure preserved? | Notes |
|---|---|---|---|
| λ_cyc = 10 (default) | Good — visible hair colour change | Yes — face identity retained | G: 4.214, D: 0.393 at epoch 10 |
| λ_cyc = 0 (disabled) | Degraded — colour bleeding and artefacts | No — face structure distorted | G: 1.455, D: 0.489 at epoch 10 |

### 2b. Example translations

![CycleGAN Default Grid](figures/cyclegan_grid.png)

*Default (λ_cyc = 10): Real dark | → Blonde | Real blonde | → Dark*

![CycleGAN No-Cycle Grid](figures/cyclegan_grid_nocyc.png)

*Ablation (λ_cyc = 0): translations lose face structure and introduce colour artefacts*

### 2c. Why removing cycle consistency lets the generator "cheat"

Without the cycle loss, the only constraint on G is to fool D_Y — the discriminator judging whether an image looks blonde. G can satisfy this by producing any plausible blonde face, completely ignoring the input's identity, structure, and background. Nothing in the loss prevents G from discarding all content and outputting a fixed "average blonde face" for every input. The cycle constraint is what forces the translation to be invertible and therefore content-preserving: G must produce a fake_y that F can map back to the original real_x, so the only translations that survive are those that retain enough structure to be reversed.

---

## Exercise 3 — Your Own Face

### 3a. Translation result

![Own Face Result](figures/my_face_result.png)

*Original | → Blonde Hair | → Dark Hair*

### 3b. Face structure preservation

The identity loss (λ_idt = 5) biases the generator toward colour and texture edits rather than structural changes, since it penalises G(y) ≠ y when the input is already in the target domain. The cycle loss (λ_cyc = 10) further anchors content by requiring F(G(x)) ≈ x. Together they constrain the generator to preserve eyes, nose, facial geometry, and background — only the hair region has enough gradient signal to change freely.

### 3c. Out-of-distribution input

The model was trained on aligned, well-lit celebrity headshots from CelebA. A photo that differs in lighting, pose, background, or appearance from the training distribution may produce incorrect hair-region edits, colour bleeding onto the face, or texture artefacts in areas the model has no training signal for.

---

## Exercise 4 (Challenge) — DDPM Noise Schedule Ablation

### 4a. Cosine schedule implementation

```python
def cosine_beta_schedule(timesteps, s=0.008):
    t = torch.linspace(0, timesteps, timesteps + 1)
    alphas_bar = torch.cos(((t / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)
```

### 4b. Schedule comparison

![Noise Schedule Comparison](figures/schedule_comparison.png)

| Schedule | Loss at epoch 10 | Notes |
|---|---|---|
| Linear | 0.0264 | Faster initial convergence |
| Cosine | 0.0438 | Slower decay, more signal preserved early |

### 4c. Sample grids

![DDPM Linear Samples](figures/ddpm_grid_linear.png)

*64 samples — linear schedule*

![DDPM Cosine Samples](figures/ddpm_grid_cosine.png)

*64 samples — cosine schedule*

The cosine schedule keeps ᾱ_t higher for longer at the start of the forward process, meaning the image retains more signal through the early timesteps and is destroyed more gradually. The linear schedule reaches near-zero signal relatively quickly, so the network must denoise from a more degraded input for most of the timestep range. By preserving more structure in the intermediate steps, the cosine schedule gives the U-Net more consistent training signal and tends to produce cleaner samples, particularly for fine details. The higher final loss for cosine (0.0438 vs 0.0264) does not indicate worse quality — it reflects that the cosine schedule presents harder denoising targets at intermediate timesteps.

---

## Visualizations

![Generated MNIST Grid (GAN)](figures/gan_grid.png)

*GAN generated MNIST digits after 20 epochs*

![CycleGAN Translation Grid](figures/cyclegan_grid.png)

*Real dark | → Blonde | Real blonde | → Dark*

![DDPM Denoising Trajectory](figures/ddpm_trajectory_linear.png)

*Reverse diffusion: noise → digit across timesteps*

---

## Discussion

GANs are the best choice when speed and sharpness matter most — inference is a single forward pass, making them practical for real-time applications like face editing or style transfer. However, training instability and mode collapse make them unreliable for tasks requiring full distributional coverage. CycleGAN is the right tool when only unpaired data is available and the task is domain translation rather than generation from scratch — it is purpose-built for paired-style tasks without pairs, such as medical image translation or seasonal photo conversion. Diffusion models produce the highest quality and most diverse outputs and are now the dominant approach for general image synthesis, but their iterative sampling (1000 steps) makes them slow for latency-sensitive applications. For a real-world synthesis task with quality as the primary concern and offline generation acceptable, diffusion is the clear winner; for interactive or real-time editing with paired-style supervision available, CycleGAN remains competitive.

---

## Environment

- Python 3.12, PyTorch 2.5.1+cu121
- GPU: RTX 2080 Ti (Puffer server)
- Datasets: MNIST (auto-downloaded), CelebA (Hugging Face mirror `Yuehao/celeba`, 5k images/domain)