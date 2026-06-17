# A3: Self-Supervised Learning

## How to Run

```bash
# Activate environment
source code/.venv/bin/activate

# Train
python3 run.py --model simclr  --epochs 10 --train
python3 run.py --model dino    --epochs 10 --train
python3 run.py --model mae     --epochs 10 --train

# Linear evaluation
python3 run.py --model simclr  --weights checkpoints/simclr.pt             --evaluate --linear
python3 run.py --model dino    --weights checkpoints/dino.pt                --evaluate --linear
python3 run.py --model mae     --weights checkpoints/mae_encoder_mask75.pt  --evaluate --linear

# DINO ablations
python3 run.py --model dino --no-centering  --epochs 10 --train
python3 run.py --model dino --n-local 0     --epochs 10 --train

# MAE mask ratio ablations
python3 run.py --model mae --mask-ratio 0.25 --epochs 5 --train
python3 run.py --model mae --mask-ratio 0.50 --epochs 5 --train
python3 run.py --model mae --mask-ratio 0.75 --epochs 5 --train
```

## Results

### Main Results

| Model | Linear Eval Acc | Time/epoch | Notes |
|---|---|---|---|
| SimCLR (ResNet-18) | 61.82% | ~78s | Contrastive, NT-Xent loss |
| DINO (default) | 44.57% | ~294s | 2 global + 4 local crops, with centering |
| DINO (no centering) | 34.02% | ~325s | Collapse ablation |
| DINO (no local crops) | 40.24% | ~131s | n_local=0 ablation |
| MAE mask=0.75 | 40.35% | ~22s | Default reconstruction |
| MAE mask=0.50 | 37.09% | ~24s | Masking ablation |
| MAE mask=0.25 | 36.37% | ~29s | Masking ablation |

---

## Exercise 1 — DINO Ablations

### 1. DINO Variant Comparison

| Setting | Linear Eval Accuracy |
|---|---|
| Default (2 global + 4 local, with centering) | 44.57% |
| No centering (`- self.center` removed) | 34.02% |
| No local crops (`n_local=0`) | 40.24% |

### 1a. Center Norm Across Epochs

| Epoch | Default DINO | No-Centering DINO |
|---|---|---|
| 1 | 4.30 | 46.76 |
| 2 | 5.94 | 61.61 |
| 3 | 7.23 | 69.83 |
| 4 | 8.55 | 78.27 |
| 5 | 9.58 | 86.36 |
| 6 | 10.82 | 96.12 |
| 7 | 11.96 | 106.28 |
| 8 | 13.39 | 116.15 |
| 9 | 14.16 | 125.46 |
| 10 | 14.58 | 136.66 |

The default DINO center norm grows in early epochs then stabilizes around 14–15, indicating the centering mechanism is actively correcting for distribution drift and reaching equilibrium. The no-centering run shows no meaningful center (nothing is being subtracted), and the unbounded growth of the raw teacher output norm (46 → 136) is a clear signature of collapse — one dominant dimension keeps growing unchecked.

### 1b. Discussion

**Why does removing centering cause collapse?**

Without centering, the teacher's softmax output can be dominated by a small number of dimensions — one "easy" direction that gives minimum loss for every input. The student learns to always output the same vector regardless of the input image, because matching that collapsed teacher distribution drives the loss to near-zero. This is confirmed by the suspiciously low loss (0.43 at epoch 10 vs 3.63 for default) and the 10.55% accuracy drop. Centering subtracts a running mean of teacher outputs, forcing the distribution to spread across all dimensions and preventing any single direction from dominating.

**Why do local crops hurt representation quality when removed?**

The multi-crop strategy forces the student to predict the teacher's global view from a small local patch — a much harder task that requires understanding global context from local information. Without local crops, the student only sees the same scale of views as the teacher, making the task easier and the learned features less rich. This is confirmed by the 4.33% accuracy drop (44.57% → 40.24%) and the much faster epoch time (~131s vs ~294s) since fewer crops are processed per step.

---

## Exercise 2 — MAE Masking Ablations

### 2. MAE Mask Ratio Comparison (5 epochs each)

| Mask Ratio | Recon Loss | Linear Eval Acc |
|---|---|---|
| 0.25 | 0.3687 | 36.37% |
| 0.50 | 0.4473 | 37.09% |
| 0.75 | 0.5315 | 40.35% |

**Why does low masking (0.25) produce worse representations even though reconstruction loss is lower?**

At low masking ratios, the reconstruction task is too easy — the model can fill in masked patches by interpolating from nearby visible patches without understanding the semantic content of the image. This means the encoder learns texture interpolation rather than high-level representations. High masking (0.75) forces the model to understand global structure and object semantics to reconstruct missing patches, driving better learned features. The results confirm this: mask=0.25 achieves the lowest reconstruction loss (0.3687) but the worst linear eval accuracy (36.37%), while mask=0.75 has the highest loss (0.5315) but the best accuracy (40.35%). Reconstruction loss and representation quality are inversely correlated at low masking ratios.

---

## Exercise 3 — Three-Way Comparison

| Metric | SimCLR | DINO | MAE |
|---|---|---|---|
| Backbone | ResNet-18 | ViT-Tiny | ViT |
| Needs negative pairs? | Yes | No | No |
| Needs EMA teacher? | No | Yes | No |
| Linear Eval Accuracy | 61.82% | 44.57% | 40.35% |
| Training time/epoch | ~78s | ~294s | ~22s |
| t-SNE cluster quality (1–5) | 3 | 3 | 2 |
| Has interpretable attention maps? | No | Yes | No |

### 3a. MAE vs DINO for large-scale pre-training

**Two reasons MAE won out over DINO for large-scale general pre-training:**

1. **Scalability and simplicity:** MAE's reconstruction objective scales more cleanly with model and data size. It requires no teacher network, no centering trick, and no carefully tuned multi-crop augmentation strategy — fewer hyperparameters and less training complexity at scale.

2. **Computational efficiency:** Because MAE's encoder only processes visible (unmasked) patches during training (25% of patches at mask_ratio=0.75), it is significantly faster and more memory-efficient per step than DINO, which must run all crops through both student and teacher networks. This gap grows with model size.

**One reason DINO is still preferred for CV-only tasks like segmentation:**

DINO's emergent attention maps produce interpretable, spatially precise features that localize foreground objects without any pixel-level supervision. The [CLS] token self-attention naturally segments objects, making DINO features directly useful for dense prediction tasks like segmentation without modification.

### 3b. Medical Image Segmentation with 500 Labeled Scans

I would choose **DINO** for this task.

With only 500 labeled scans, the most critical property of a pre-training method is whether its features transfer well to dense spatial tasks like segmentation. DINO's ViT produces spatially structured attention maps that naturally localize anatomical structures — this emergent object segmentation property is particularly valuable for medical imaging, where anatomical structure is consistent across patients. The attention map interpretability also matters in clinical settings where model decisions must be explainable. MAE would require significantly longer pre-training to achieve competitive spatial features, and SimCLR's contrastive features are optimized for image-level classification rather than spatial structure. DINO gives the best combination of spatial feature quality, interpretability, and data efficiency for a 500-scan limited-label scenario.

---

## Environment

- Python 3.12, PyTorch 2.5.1+cu121
- GPU: RTX 2080 Ti (Puffer server)
- Dataset: CIFAR-10 (auto-downloaded via torchvision)