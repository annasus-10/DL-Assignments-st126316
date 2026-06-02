# A1: Representation Learning — Teaching Guide

A walkthrough of the notebook, cell by cell, with the story behind each architecture and what to say to students.

---

## Opening (Cell 0–2): Setting the Stage

**What to say:**

> "Today we're going to trace 10 years of computer vision history — from AlexNet in 2012 to Vision Transformers in 2020. Each model we look at was solving a specific problem with the previous one. By the end, you should be able to explain *why* each architecture exists, not just *what* it does."

Before diving into models, Cell 1–2 covers PyTorch basics: tensors, kernels, channels, parameter counting.

**Pause and ask students:**
- "If a Conv2d layer has kernel_size=3, in_channels=3, out_channels=64 — how many parameters does it have?"
- Answer: `3 × 3 × 3 × 64 + 64 (bias) = 1,792`

---

## Part 1: AlexNet (Cells 3–28)

### The story (tell this before showing any code)

**The problem before 2012:**

Computer vision was dominated by hand-crafted features — SIFT, HOG, etc. People spent years manually designing feature extractors. Neural networks existed but were considered too slow and too hard to train for images.

**What AlexNet did in 2012:**

AlexNet won ImageNet (ILSVRC 2012) with **top-5 error of 15.3%** — the second place was 26.2%. That gap was so shocking it ended the hand-crafted feature era overnight.

Key ingredients:
| Ingredient | Why it mattered |
|-----------|----------------|
| **ReLU** instead of sigmoid/tanh | No vanishing gradient in activation; trains ~6x faster |
| **Dropout (0.5)** | First time used at scale — prevents overfitting |
| **Data augmentation** | Random crops, flips — effectively multiplied dataset size |
| **GPU training** | Two GTX 580s — made deep networks practical for the first time |
| **Local Response Normalization** | Lateral inhibition (controversial — not used in modern nets) |

**The meme:** AlexNet is basically "what if we take LeNet from 1998 and make it 100x bigger and train on a GPU?" The answer turned out to be: it works incredibly well.

### Sequential API vs Module API (Cells 11–25)

**What to say:**

> "PyTorch gives you two ways to build a model. Sequential is like LEGO — stack layers in a line. Module is like writing a class — you control exactly what happens in `forward()`. AlexNet is simple enough for Sequential, but GoogLeNet has branching paths, so we need Module."

Show students the difference:
```python
# Sequential: simple, no branching
model = nn.Sequential(nn.Conv2d(...), nn.ReLU(), nn.Linear(...))

# Module: full control
class AlexNet(nn.Module):
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

**Key point:** From GoogLeNet onwards, every model requires Module because they have non-linear data flows.

---

## Part 2: GoogLeNet / Inception (Cells 29–38)

### The story

**The problem with AlexNet:** It just stacks convolutions deeper. People tried to make it deeper — they kept hitting two walls:

1. **Overfitting** — more layers, more parameters, not enough data
2. **Computational cost** — 3×3 convolutions get expensive fast

**The Inception insight (2014):**

> "What if instead of choosing ONE kernel size, we use ALL of them at the same time?"

```
Input
 ├── 1×1 Conv  (capture pointwise features)
 ├── 3×3 Conv  (capture local patterns)
 ├── 5×5 Conv  (capture larger patterns)
 └── 3×3 MaxPool (capture spatial hierarchy)
      ↓
   Concatenate all outputs along channel dimension
```

**The meme — "We need to go deeper" (Inception the movie):**

The name "Inception" literally comes from the movie. The original paper's title is *"Going Deeper with Convolutions."* The joke is that inside each Inception module, you're doing multiple convolutions simultaneously — a network within a network.

**Why it's better than AlexNet:**

| | AlexNet | GoogLeNet |
|--|---------|-----------|
| Parameters | 60M | 5M |
| Depth | 8 layers | 22 layers |
| Top-5 error (ImageNet) | 15.3% | 6.7% |

12x fewer parameters, 3x deeper, much more accurate. The secret: **1×1 convolutions** act as a bottleneck that reduces channels before the expensive 3×3 and 5×5 convolutions.

**What to say when showing code:**

> "Look at the Inception module code. Notice the 1×1 conv before the 3×3 and 5×5. That's not for learning features — it's a dimension reduction trick. If input is 256 channels and we reduce to 64 first, the 3×3 conv costs 64×64×3×3 instead of 256×256×3×3. That's a 16x reduction in computation."

---

## Part 3: ResNet (Cells 39–46)

### The story

**The problem that nobody expected:**

After GoogLeNet, people thought: "deeper = better, let's just keep going." They tried 20-layer, 30-layer, 56-layer networks.

**Shocking result:** A 56-layer plain network performed *worse* than a 20-layer network — **not just on test set, but on training set too.**

This is not overfitting. Overfitting means train accuracy goes up but test accuracy goes down. Here both went down. This is a fundamentally different problem.

**He et al.'s diagnosis (2015):**

> "The problem is optimization. Very deep networks are hard to train. Gradients vanish before they reach the early layers."

**The gradient problem visualized:**

```
Loss
 ↓ ×0.9   (layer 50)
 ↓ ×0.9   (layer 49)
 ↓ ×0.9   ...
 ↓ ×0.9   (layer 1)
 
After 50 layers: 0.9^50 = 0.005  → gradient is basically zero
Early layers learn nothing.
```

**The ResNet solution — Skip Connections:**

> "What if we let the gradient take a shortcut?"

```python
def forward(self, x):
    residual = x                          # save input
    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out += residual                       # add original input back
    out = F.relu(out)
    return out
```

The skip connection means the network learns **F(x) + x** instead of **F(x)**.

**Why this helps:**
- Gradient can flow directly through the `+` operation without passing through convolutions
- If the convolutions learn nothing useful (F(x) ≈ 0), the block becomes identity: output = input
- The network can effectively "skip" layers it doesn't need → easier to optimize

**Key insight to hammer home:**

> "ResNet doesn't make each layer smarter. It makes each layer *optional*. The network can choose to use a layer or bypass it. This is why you can train ResNet-152 (152 layers!) and it still converges."

**Results:**
| Model | Year | ImageNet top-5 error |
|-------|------|---------------------|
| AlexNet | 2012 | 15.3% |
| GoogLeNet | 2014 | 6.7% |
| ResNet-152 | 2015 | **3.57%** |
| Human | — | ~5% |

ResNet-152 beat human performance on ImageNet.

---

## Part 4: Vision Transformer — ViT (Cells 47–55)

### The story

**Context — 2017–2020:**

Transformers (Attention is All You Need, 2017) took over NLP completely. BERT, GPT — all Transformers. Meanwhile, CNNs still ruled computer vision.

**The question researchers asked:**

> "Can we apply a Transformer *directly* to images with no convolutions at all?"

Previous attempts had always mixed CNNs with attention. ViT (2020) went all-in: **zero convolutions.**

### The Patch idea

**What to say:**

> "A Transformer processes a sequence of tokens. In NLP, tokens are words. What's the equivalent of a word in an image?"

**The answer: patches.**

```
32×32 image
 → split into 4×4 patches
 → 64 patches total
 → flatten each patch: 4×4×3 = 48 numbers
 → project to embedding dimension (e.g. 128)
 → treat 64 embeddings as a sequence
 → add [CLS] token at the front
 → add positional embeddings
 → feed into Transformer encoder
 → take [CLS] output → classify
```

**Show this in code:**
```python
# This one line does all the patch work:
self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
# A conv with kernel=stride=patch_size tiles exactly into non-overlapping patches
```

### Why ViT wins (but only with enough data)

**The catch — ViT needs massive data:**

CNNs have **inductive biases** built in:
- **Translation invariance** — a cat in the top-left and bottom-right both activate the same filter
- **Local connectivity** — each neuron only looks at a small region

ViT has **none of these biases**. Every patch attends to every other patch from layer 1. This is more flexible but requires much more data to learn these patterns from scratch.

| Training data | CNN (ResNet) | ViT |
|--------------|-------------|-----|
| Small (CIFAR-10) | Strong | Weak |
| Medium (ImageNet 1M) | Strong | Comparable |
| Large (JFT 300M+) | Good | **ViT wins** |

**Why ViT ultimately wins:**

> "CNNs assume locality — nearby pixels relate to each other. This is usually true, but it's a constraint. ViT makes no such assumption — it can learn that a patch in the top-left relates to a patch in the bottom-right. With enough data, this flexibility becomes a superpower."

Global context from layer 1: ViT can model long-range dependencies that would require many CNN layers to capture.

**The modern picture:**

ViT started a paradigm shift. Today the best vision models (DINOv2, SAM, CLIP) are all ViT-based. CNNs haven't disappeared but they're no longer the default first choice for new research.

---

## The Big Picture — Timeline to tell at the end

```
2012  AlexNet      "Deep CNNs work. GPUs matter. ReLU + Dropout."
        ↓
2014  GoogLeNet    "Go wider, not just deeper. Multi-scale features. 1×1 bottleneck."
        ↓
2015  ResNet       "Skip connections let gradients flow. Now we can go 100+ layers deep."
        ↓
2020  ViT          "No convolutions needed. Patches + Transformer. Scales with data."
        ↓
2022+ Foundation   CLIP, DINOv2, SAM — ViT trained on billions of images,
      Models       zero-shot transfer to anything.
```

**Closing question for students:**

> "If you had to train a model on 500 labeled images of a rare disease, which architecture would you choose and why?"
>
> Answer: Pretrained ResNet or ViT (fine-tuned) — not from scratch. With only 500 images, inductive bias from pretraining matters more than architecture flexibility.

---

## Exercise Walkthrough Tips

**Ex 1 Q2 (LRN):** Ask students — "Do modern networks use LRN?" (No — BatchNorm replaced it entirely by 2015)

**Ex 1 Q4 (AlexNet vs GoogLeNet comparison):** Tell them to focus on the parameter count difference. GoogLeNet has 12x fewer parameters but is better — this was the moment researchers learned that *more parameters ≠ better model*.

**Ex 1 Q6 (ResNet two-stage fine-tuning):** Use the "new employee" analogy — don't let a new hire change how the whole experienced team works on day 1. Let them learn the job first, then adjust together.

**Ex 2 (ViT fine-tuning):** Warn them — ViT-B/16 needs 224×224 input. CIFAR-10 is 32×32. They need to resize. This is a common gotcha.
