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
python3 run.py --model simclr  --weights code/checkpoints/simclr.pt        --evaluate --linear
python3 run.py --model dino    --weights code/checkpoints/dino.pt           --evaluate --linear
python3 run.py --model mae     --weights code/checkpoints/mae_encoder.pt    --evaluate --linear

# DINO ablations
python3 run.py --model dino --no-centering  --epochs 10 --train
python3 run.py --model dino --n-local 0     --epochs 10 --train

# MAE mask ratio ablations
python3 run.py --model mae --mask-ratio 0.25 --epochs 5 --train
python3 run.py --model mae --mask-ratio 0.50 --epochs 5 --train
python3 run.py --model mae --mask-ratio 0.75 --epochs 5 --train
```

---

## Results

### Main Results Table

| Model | Linear Eval Acc | Time/epoch | Notes |
|---|---|---|---|
| SimCLR (ResNet-18) | TODO | TODO s | Contrastive, NT-Xent loss |
| DINO (default) | TODO | TODO s | 2 global + 4 local crops, with centering |
| DINO (no centering) | TODO | TODO s | Collapse ablation |
| DINO (no local crops) | TODO | TODO s | n_local=0 ablation |
| MAE mask=0.75 | TODO | TODO s | Default reconstruction |
| MAE mask=0.50 | TODO | TODO s | Masking ablation |
| MAE mask=0.25 | TODO | TODO s | Masking ablation |

---

## Exercise 1 — DINO Ablations

### 1. DINO Variant Comparison

| Setting | Linear Eval Accuracy |
|---|---|
| Default (2 global + 4 local, with centering) | TODO |
| No centering (`- self.center` removed) | TODO |
| No local crops (`n_local=0`) | TODO |

### 1a. Center Norm Plot

<!-- Insert saved/dino_center_norm.png here -->

**Observation:** The center norm TODO (grows / shrinks / stabilizes) across training epochs. TODO: describe what you see in 1–2 sentences.

### 1b. Discussion

**Why does removing centering cause collapse?**

Without centering, the teacher's softmax output can be dominated by a small number of dimensions — one "easy" direction that gives minimum loss for every input. The student learns to always output the same vector regardless of the input image, because matching that collapsed teacher distribution drives the loss to near-zero. Centering subtracts a running mean of teacher outputs, forcing the distribution to spread across all dimensions and preventing any single direction from dominating.

**Why do local crops hurt representation quality when removed?**

The multi-crop strategy forces the student to predict a teacher's global view from a small local patch — a much harder task that requires understanding global context from local information. Without local crops, the student only sees the same scale of views as the teacher, making the task easier and the learned features less rich. Expect ~2–5% lower linear eval accuracy when n_local=0.

---

## Exercise 2 — MAE Masking Ablations

### 2. MAE Mask Ratio Comparison (5 epochs each)

| Mask Ratio | Recon Loss | Linear Eval Acc |
|---|---|---|
| 0.25 | TODO | TODO |
| 0.50 | TODO | TODO |
| 0.75 | TODO | TODO |

**Why does low masking (0.25) produce worse representations even though reconstruction loss is lower?**

At low masking ratios, the reconstruction task is too easy — the model can fill in masked patches by interpolating from nearby visible patches without understanding the semantic content of the image. This means the encoder learns texture interpolation rather than high-level representations. High masking (0.75) forces the model to understand global structure and object semantics to reconstruct missing patches, driving better learned features. The key insight is that **reconstruction loss and representation quality are inversely correlated at low masking ratios**: lower loss means easier task means weaker encoder.

---

## Exercise 3 — Three-Way Comparison

| Metric | SimCLR | DINO | MAE |
|---|---|---|---|
| Backbone | ResNet-18 | ViT-Tiny | ViT |
| Needs negative pairs? | Yes | No | No |
| Needs EMA teacher? | No | Yes | No |
| Linear Eval Accuracy | TODO | TODO | TODO |
| Training time/epoch | TODO s | TODO s | TODO s |
| t-SNE cluster quality (1–5) | TODO | TODO | TODO |
| Has interpretable attention maps? | No | Yes | No |

### 3a. MAE vs DINO for large-scale pre-training

**Two reasons MAE won out over DINO for large-scale general pre-training:**

1. **Scalability with fewer assumptions:** MAE's reconstruction objective scales more cleanly with model and data size. It requires no teacher network, no centering trick, and no carefully tuned multi-crop augmentation strategy — fewer hyperparameters and less training complexity.

2. **Computational efficiency:** Because MAE's encoder processes only visible (unmasked) patches during training (typically 25% of patches at mask_ratio=0.75), it is significantly faster and more memory-efficient per step than DINO, which must run all crops through both student and teacher networks.

**One reason DINO is still preferred for CV-only tasks like segmentation:**

DINO's emergent attention maps produce interpretable, spatially precise features that localize foreground objects without any pixel-level supervision. The [CLS] token self-attention naturally segments objects, making DINO features directly useful for dense prediction tasks like segmentation without modification.

### 3b. Medical Image Segmentation with 500 Labeled Scans

I would choose **DINO** for this task.

With only 500 labeled scans, the most critical property of a pre-training method is whether its features transfer well to dense spatial tasks like segmentation. DINO's ViT produces spatially structured attention maps that naturally localize anatomical structures — this emergent object segmentation property is particularly valuable for medical imaging, where anatomical structure is consistent across patients. The attention map interpretability also matters in clinical settings where model decisions must be explainable. MAE would require significantly longer pre-training (hundreds of epochs) to achieve competitive spatial features, and SimCLR's contrastive features are optimized for image-level classification rather than spatial structure. DINO gives the best combination of spatial feature quality, interpretability, and data efficiency for a 500-scan limited-label scenario.

---

## Visualizations

### Loss Curves

<!-- Insert loss_curves.png here -->

### DINO Attention Maps

<!-- Insert dino_attention_maps.png here -->

*DINO [CLS] token self-attention across all heads for 5 test images. Note emergent foreground segmentation — no segmentation labels were used during training.*

### MAE Reconstruction Grid

<!-- Insert mae_reconstruction.png here -->

*MAE reconstruction: original → masked (75%) → reconstructed. The model reconstructs semantically plausible content despite 75% masking.*

### t-SNE Comparison

<!-- Insert tsne_comparison.png here -->

*t-SNE projection of 2000 test embeddings per model. Well-separated clusters indicate semantically meaningful representations.*

---

## Environment

- Python 3.12, PyTorch 2.5.1+cu121
- GPU: RTX 2080 Ti (Puffer server)
- CIFAR-10 dataset (auto-downloaded)