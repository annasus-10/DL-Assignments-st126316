import argparse
import os
import time
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from PIL import Image
from sklearn.manifold import TSNE


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

EVAL_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

MAE_MEAN = [0.4914, 0.4822, 0.4465]
MAE_STD  = [0.247,  0.243,  0.261]


# ─── Datasets ─────────────────────────────────────────────────────────────────

class CIFAR10SSL(Dataset):
    """SimCLR dataset — returns two augmented views per image."""
    def __init__(self, root='data', train=True, augmentation=None):
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        self.augment = augmentation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        x_i, x_j = self.augment(img)
        return x_i, x_j, label


class CIFAR10DINO(Dataset):
    """DINO dataset — returns multi-crop views per image."""
    def __init__(self, root='data', train=True, n_local=4):
        from models.dino import DINOAugmentation
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        self.augment = DINOAugmentation(n_local=n_local)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.augment(img), label


def dino_collate(batch):
    crops_list, labels = zip(*batch)
    n_views = len(crops_list[0])
    stacked = [torch.stack([crops_list[i][v] for i in range(len(crops_list))]) for v in range(n_views)]
    return stacked, torch.tensor(labels)


# ─── Linear Evaluation ────────────────────────────────────────────────────────

def linear_eval(encoder_fn, embed_dim, device, data_root='data', mean=None, std=None, epochs=10):
    """Train a linear classifier on frozen encoder features."""
    if mean is None:
        mean = [0.4914, 0.4822, 0.4465]
    if std is None:
        std = [0.2023, 0.1994, 0.2010]

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_ds = torchvision.datasets.CIFAR10(data_root, train=True,  transform=train_tf, download=True)
    test_ds  = torchvision.datasets.CIFAR10(data_root, train=False, transform=test_tf,  download=True)
    trl = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=2)
    tel = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2)

    clf     = nn.Linear(embed_dim, 10).to(device)
    opt_clf = torch.optim.Adam(clf.parameters(), lr=1e-3)

    for epoch in range(epochs):
        clf.train()
        correct = total = 0
        for imgs, labels in tqdm(trl, desc=f'  Linear Eval {epoch+1}/{epochs}'):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                h = encoder_fn(imgs)
            logits = clf(h)
            loss = F.cross_entropy(logits, labels)
            opt_clf.zero_grad()
            loss.backward()
            opt_clf.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)
        print(f'  Train Acc: {correct/total*100:.2f}%')

    clf.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in tel:
            imgs, labels = imgs.to(device), labels.to(device)
            h = encoder_fn(imgs)
            correct += (clf(h).argmax(1) == labels).sum().item()
            total   += labels.size(0)
    acc = correct / total * 100
    print(f'\n✅ Linear Eval Test Accuracy: {acc:.2f}%')
    return acc


# ─── SimCLR ───────────────────────────────────────────────────────────────────

def train_simclr(args, device):
    from models.simclr import SimCLR, NTXentLoss, SimCLRAugmentation

    train_ds    = CIFAR10SSL(root=args.data_root, augmentation=SimCLRAugmentation())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, drop_last=True)

    model     = SimCLR().to(device)
    criterion = NTXentLoss(temperature=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    losses      = []
    epoch_times = []

    for epoch in range(args.epochs):
        model.train()
        ep = []
        t0 = time.time()
        for x_i, x_j, _ in tqdm(train_loader, desc=f'SimCLR {epoch+1}/{args.epochs}'):
            x_i, x_j = x_i.to(device), x_j.to(device)
            z_i, z_j, _, _ = model(x_i, x_j)
            loss = criterion(z_i, z_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep.append(loss.item())
        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        losses.append(np.mean(ep))
        print(f'Epoch {epoch+1:02d} | Loss: {np.mean(ep):.4f} | Time: {elapsed:.1f}s')

    print(f'\nAvg/epoch: {np.mean(epoch_times):.1f}s')
    save_path = os.path.join(args.checkpoint_dir, 'simclr.pt')
    torch.save(model.state_dict(), save_path)
    print(f'Saved to {save_path}')


def evaluate_simclr(args, device):
    from models.simclr import SimCLR

    model = SimCLR().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    for p in model.encoder.parameters():
        p.requires_grad = False

    def encoder_fn(imgs):
        return torch.flatten(model.encoder(imgs), 1)

    linear_eval(encoder_fn, embed_dim=512, device=device, data_root=args.data_root)


# ─── DINO ─────────────────────────────────────────────────────────────────────

def train_dino(args, device):
    from models.dino import build_dino_model, DINOLoss

    OUT_DIM = 256
    EMA_M   = 0.996

    dino_dataset = CIFAR10DINO(root=args.data_root, n_local=args.n_local)
    dino_loader  = DataLoader(dino_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, drop_last=True, collate_fn=dino_collate)

    student_vit, student_head = build_dino_model(out_dim=OUT_DIM)
    teacher_vit, teacher_head = build_dino_model(out_dim=OUT_DIM)
    student_vit,  student_head  = student_vit.to(device),  student_head.to(device)
    teacher_vit,  teacher_head  = teacher_vit.to(device),  teacher_head.to(device)
    teacher_vit.load_state_dict(student_vit.state_dict())
    teacher_head.load_state_dict(student_head.state_dict())
    for p in teacher_vit.parameters():  p.requires_grad = False
    for p in teacher_head.parameters(): p.requires_grad = False

    use_centering = not args.no_centering
    dino_loss_fn  = DINOLoss(out_dim=OUT_DIM, use_centering=use_centering).to(device)
    optimizer_d   = torch.optim.AdamW(
        list(student_vit.parameters()) + list(student_head.parameters()),
        lr=args.lr, weight_decay=0.04
    )

    losses      = []
    epoch_times = []
    center_norms = []

    for epoch in range(args.epochs):
        student_vit.train()
        student_head.train()
        ep = []
        t0 = time.time()
        for crops, _ in tqdm(dino_loader, desc=f'DINO {epoch+1}/{args.epochs}'):
            crops = [c.to(device) for c in crops]
            student_out = [student_head(student_vit(c)) for c in crops]
            with torch.no_grad():
                teacher_out = [teacher_head(teacher_vit(crops[0])),
                               teacher_head(teacher_vit(crops[1]))]
            loss = dino_loss_fn(student_out, teacher_out)
            optimizer_d.zero_grad()
            loss.backward()
            optimizer_d.step()
            with torch.no_grad():
                for s_p, t_p in zip(student_vit.parameters(), teacher_vit.parameters()):
                    t_p.data = EMA_M * t_p.data + (1 - EMA_M) * s_p.data
                for s_p, t_p in zip(student_head.parameters(), teacher_head.parameters()):
                    t_p.data = EMA_M * t_p.data + (1 - EMA_M) * s_p.data
            ep.append(loss.item())

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        losses.append(np.mean(ep))
        center_norm = dino_loss_fn.center.norm().item()
        center_norms.append(center_norm)
        print(f'Epoch {epoch+1:02d} | Loss: {np.mean(ep):.4f} | Center norm: {center_norm:.4f} | Time: {elapsed:.1f}s')

    print(f'\nAvg/epoch: {np.mean(epoch_times):.1f}s')

    suffix = ''
    if args.no_centering:
        suffix = '_no_centering'
    elif args.n_local == 0:
        suffix = '_no_local'

    save_path = os.path.join(args.checkpoint_dir, f'dino{suffix}.pt')
    torch.save({
        'student_vit':  student_vit.state_dict(),
        'student_head': student_head.state_dict(),
        'center_norms': center_norms,
    }, save_path)
    print(f'Saved to {save_path}')


def evaluate_dino(args, device):
    from models.dino import build_dino_model

    student_vit, _ = build_dino_model(out_dim=256)
    student_vit = student_vit.to(device)
    ckpt = torch.load(args.weights, map_location=device)
    student_vit.load_state_dict(ckpt['student_vit'])
    for p in student_vit.parameters():
        p.requires_grad = False

    def encoder_fn(imgs):
        return student_vit(imgs)

    linear_eval(encoder_fn, embed_dim=student_vit.embed_dim, device=device, data_root=args.data_root)


# ─── MAE ──────────────────────────────────────────────────────────────────────

def train_mae(args, device):
    from models.mae import MAE

    mae_train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MAE_MEAN, MAE_STD),
    ])
    train_ds    = torchvision.datasets.CIFAR10(args.data_root, train=True,
                                               transform=mae_train_tf, download=True)
    mae_loader  = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=2, pin_memory=True, drop_last=True)

    model = MAE(mask_ratio=args.mask_ratio).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.05, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    losses      = []
    epoch_times = []

    for epoch in range(args.epochs):
        model.train()
        ep = []
        t0 = time.time()
        for imgs, _ in tqdm(mae_loader, desc=f'MAE {epoch+1}/{args.epochs}'):
            imgs = imgs.to(device)
            loss, _, _ = model(imgs)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep.append(loss.item())
        scheduler.step()
        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        losses.append(np.mean(ep))
        print(f'Epoch {epoch+1:02d} | Recon Loss: {np.mean(ep):.4f} | Time: {elapsed:.1f}s')

    print(f'\nAvg/epoch: {np.mean(epoch_times):.1f}s')

    suffix    = f'_mask{int(args.mask_ratio*100)}'
    save_path = os.path.join(args.checkpoint_dir, f'mae_encoder{suffix}.pt')
    torch.save(model.encoder.state_dict(), save_path)
    print(f'Saved to {save_path}')


def evaluate_mae(args, device):
    from models.mae import MAE

    model = MAE(mask_ratio=args.mask_ratio).to(device)
    model.encoder.load_state_dict(torch.load(args.weights, map_location=device))
    model.encoder.eval()
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.encoder.mask_ratio = 0.0

    def encoder_fn(imgs):
        x_vis, _, _ = model.encoder(imgs)
        return x_vis.mean(dim=1)

    linear_eval(encoder_fn, embed_dim=model.encoder.embed_dim,
                device=device, data_root=args.data_root,
                mean=MAE_MEAN, std=MAE_STD)

# ─── Visualizations ───────────────────────────────────────────────────────────

def visualize_dino(args, device):
    import matplotlib.pyplot as plt
    from models.dino import build_dino_model

    os.makedirs('figures', exist_ok=True)

    student_vit, _ = build_dino_model(out_dim=256)
    student_vit = student_vit.to(device)
    ckpt = torch.load(args.weights, map_location=device)
    student_vit.load_state_dict(ckpt['student_vit'])
    student_vit.eval()

    # ── Attention maps ────────────────────────────────────────────────────────
    img_mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    img_std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

    attentions = {}
    attn_module = student_vit.blocks[-1].attn
    original_forward = attn_module.forward

    def patched_attn_forward(x, **kwargs):
        B, N, C = x.shape
        qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads,
                                          C // attn_module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn_w = (q @ k.transpose(-2, -1)) * attn_module.scale
        attn_w = attn_w.softmax(dim=-1)
        attentions['last'] = attn_w.detach()
        attn_w = attn_module.attn_drop(attn_w)
        x = (attn_w @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_module.proj(x)
        x = attn_module.proj_drop(x)
        return x

    attn_module.forward = patched_attn_forward

    raw_test = torchvision.datasets.CIFAR10('data', train=False, transform=EVAL_TF)
    img_loader = DataLoader(raw_test, batch_size=1, shuffle=True)
    n_heads = student_vit.blocks[-1].attn.num_heads
    patch_h = patch_w = 32 // 4

    fig, axes = plt.subplots(5, n_heads + 1, figsize=(2 * (n_heads + 1), 12))
    sample_iter = iter(img_loader)

    for row in range(5):
        img_tensor, label = next(sample_iter)
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            _ = student_vit(img_tensor)
        attn = attentions['last']
        cls_attn = attn[0, :, 0, 1:]
        img_disp = torch.clamp(img_tensor[0].cpu() * img_std + img_mean, 0, 1).permute(1, 2, 0).numpy()
        axes[row][0].imshow(img_disp)
        axes[row][0].set_title(CLASSES[label.item()], fontsize=9)
        axes[row][0].axis('off')
        for h in range(n_heads):
            head_map = cls_attn[h].reshape(patch_h, patch_w).cpu().numpy()
            head_map = (head_map - head_map.min()) / (head_map.max() - head_map.min() + 1e-8)
            head_up = np.array(Image.fromarray((head_map * 255).astype(np.uint8)).resize((32, 32)))
            axes[row][h + 1].imshow(img_disp, alpha=0.4)
            axes[row][h + 1].imshow(head_up, cmap='hot', alpha=0.7, vmin=0, vmax=255)
            if row == 0:
                axes[row][h + 1].set_title(f'Head {h+1}', fontsize=8)
            axes[row][h + 1].axis('off')

    plt.suptitle('DINO Self-Attention Maps: [CLS] token → patches\n'
                 'Emergent object segmentation — no segmentation labels used!',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig('figures/dino_attention_maps.png', dpi=120, bbox_inches='tight')
    plt.close()
    print('Saved figures/dino_attention_maps.png')

    # ── Center norm plot ──────────────────────────────────────────────────────
    if 'center_norms' in ckpt:
        center_norms = ckpt['center_norms']
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(1, len(center_norms) + 1), center_norms, marker='o', color='darkorange')
        ax.set_title('DINO Center Norm Across Epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('center.norm()')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig('figures/dino_center_norm.png', dpi=120, bbox_inches='tight')
        plt.close()
        print('Saved figures/dino_center_norm.png')

    attn_module.forward = original_forward


def visualize_mae(args, device):
    import matplotlib.pyplot as plt
    from models.mae import MAE

    os.makedirs('figures', exist_ok=True)

    mae_test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MAE_MEAN, MAE_STD),
    ])

    model = MAE(mask_ratio=args.mask_ratio).to(device)
    model.encoder.load_state_dict(torch.load(args.weights, map_location=device))
    model.encoder.mask_ratio = 0.75
    model.eval()

    imgs_viz, _ = next(iter(DataLoader(
        torchvision.datasets.CIFAR10('data', train=False, transform=mae_test_tf),
        batch_size=8, shuffle=True
    )))
    imgs_viz = imgs_viz.to(device)

    with torch.no_grad():
        loss_viz, pred, mask = model(imgs_viz)

    p    = model.patch_size
    h_g  = w_g = 32 // p
    mean_t = torch.tensor(MAE_MEAN).view(3, 1, 1)
    std_t  = torch.tensor(MAE_STD).view(3, 1, 1)

    def unpatchify(patches, p, h, w, in_ch=3):
        N = patches.size(0)
        x = patches.reshape(N, h, w, p, p, in_ch)
        x = x.permute(0, 5, 1, 3, 2, 4)
        return x.reshape(N, in_ch, h * p, w * p)

    pred_imgs = unpatchify(pred.cpu(), p, h_g, w_g)
    orig_np   = (imgs_viz.cpu() * std_t + mean_t).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    pred_np   = (pred_imgs       * std_t + mean_t).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    mask_exp  = mask.cpu().view(-1, h_g, w_g).unsqueeze(1)
    mask_exp  = mask_exp.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)
    mask_np   = mask_exp.expand(-1, 3, -1, -1).permute(0, 2, 3, 1).numpy()
    masked_np = orig_np.copy()
    masked_np[mask_np.astype(bool)] = 0.5

    N_show = 4
    fig, axes = plt.subplots(3, N_show, figsize=(2 * N_show, 6))
    for row, (imgs_row, title) in enumerate(zip(
        [orig_np, masked_np, pred_np],
        ['Original', 'Masked (75%)', 'Reconstructed']
    )):
        axes[row, 0].set_ylabel(title, fontsize=10)
        for col in range(N_show):
            axes[row, col].imshow(imgs_row[col])
            axes[row, col].axis('off')
    plt.suptitle('MAE Reconstruction (CIFAR-10)', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/mae_reconstruction.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved figures/mae_reconstruction.png | Recon loss: {loss_viz.item():.4f}')


def visualize_tsne(args, device):
    import matplotlib.pyplot as plt
    from models.simclr import SimCLR
    from models.dino import build_dino_model
    from models.mae import MAE

    os.makedirs('figures', exist_ok=True)

    mae_test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MAE_MEAN, MAE_STD),
    ])
    test_lbl    = torchvision.datasets.CIFAR10('data', train=False, transform=EVAL_TF, download=True)
    test_lbl_mae = torchvision.datasets.CIFAR10('data', train=False, transform=mae_test_tf, download=True)
    tel     = DataLoader(test_lbl,     batch_size=256, shuffle=False, num_workers=2)
    tel_mae = DataLoader(test_lbl_mae, batch_size=256, shuffle=False, num_workers=2)

    # SimCLR embeddings
    simclr = SimCLR().to(device)
    simclr.load_state_dict(torch.load('checkpoints/simclr.pt', map_location=device))
    simclr.eval()
    simclr_emb, simclr_lbl = [], []
    with torch.no_grad():
        for imgs, labels in tel:
            imgs = imgs.to(device)
            h = torch.flatten(simclr.encoder(imgs), 1)
            simclr_emb.append(h.cpu()); simclr_lbl.append(labels)
    simclr_emb = torch.cat(simclr_emb)
    simclr_lbl = torch.cat(simclr_lbl)

    # DINO embeddings
    dino_vit, _ = build_dino_model(out_dim=256)
    dino_vit = dino_vit.to(device)
    ckpt = torch.load('checkpoints/dino.pt', map_location=device)
    dino_vit.load_state_dict(ckpt['student_vit'])
    dino_vit.eval()
    dino_emb, dino_lbl = [], []
    with torch.no_grad():
        for imgs, labels in tel:
            imgs = imgs.to(device)
            h = dino_vit(imgs)
            dino_emb.append(h.cpu()); dino_lbl.append(labels)
    dino_emb = torch.cat(dino_emb)
    dino_lbl = torch.cat(dino_lbl)

    # MAE embeddings
    mae_model = MAE(mask_ratio=0.75).to(device)
    mae_model.encoder.load_state_dict(torch.load('checkpoints/mae_encoder_mask75.pt', map_location=device))
    mae_model.encoder.mask_ratio = 0.0
    mae_model.eval()
    mae_emb, mae_lbl = [], []
    with torch.no_grad():
        for imgs, labels in tel_mae:
            imgs = imgs.to(device)
            x_vis, _, _ = mae_model.encoder(imgs)
            feats = x_vis.mean(dim=1)
            mae_emb.append(feats.cpu()); mae_lbl.append(labels)
    mae_emb = torch.cat(mae_emb)
    mae_lbl = torch.cat(mae_lbl)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for ax, (name, emb, lbls) in zip(axes, [
        ('SimCLR (ResNet-18)', simclr_emb, simclr_lbl),
        ('DINO (ViT-Tiny)',    dino_emb,   dino_lbl),
        ('MAE (ViT)',          mae_emb,    mae_lbl),
    ]):
        idx  = np.random.choice(len(emb), 2000, replace=False)
        proj = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(emb[idx].numpy())
        for c in range(10):
            mask_c = lbls[idx].numpy() == c
            ax.scatter(proj[mask_c, 0], proj[mask_c, 1], c=[colors[c]],
                       label=CLASSES[c], alpha=0.6, s=10)
        ax.set_title(name, fontsize=12)
        ax.legend(fontsize=7, markerscale=2)
        ax.axis('off')
    plt.suptitle('t-SNE: Learned Representations on CIFAR-10 (no labels used in training)', fontsize=13)
    plt.tight_layout()
    plt.savefig('figures/tsne_comparison.png', dpi=120, bbox_inches='tight')
    plt.close()
    print('Saved figures/tsne_comparison.png')
    
# ─── Argument Parser ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='A3: Self-Supervised Learning')

    parser.add_argument('--model', type=str, required=True,
                        choices=['simclr', 'dino', 'mae', 'tsne'])
    parser.add_argument('--train',          action='store_true')
    parser.add_argument('--evaluate',       action='store_true')
    parser.add_argument('--linear',         action='store_true')
    parser.add_argument('--visualize',      action='store_true')
    parser.add_argument('--epochs',         type=int,   default=10)
    parser.add_argument('--batch-size',     type=int,   default=None)
    parser.add_argument('--lr',             type=float, default=None)
    parser.add_argument('--weights',        type=str,   default=None)
    parser.add_argument('--data-root',      type=str,   default='data')
    parser.add_argument('--checkpoint-dir', type=str,   default='checkpoints')

    # DINO ablations
    parser.add_argument('--no-centering',   action='store_true')
    parser.add_argument('--n-local',        type=int,   default=4)

    # MAE ablation
    parser.add_argument('--mask-ratio',     type=float, default=0.75)

    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(42)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.data_root,      exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Set model-specific defaults if not overridden
    if args.batch_size is None:
        args.batch_size = {'simclr': 256, 'dino': 64, 'mae': 128, 'tsne': 256}[args.model]
    if args.lr is None:
        args.lr = {'simclr': 3e-4, 'dino': 5e-4, 'mae': 1.5e-4, 'tsne': 3e-4}[args.model]

    if args.train:
        if args.model == 'simclr':
            train_simclr(args, device)
        elif args.model == 'dino':
            train_dino(args, device)
        elif args.model == 'mae':
            train_mae(args, device)

    if args.evaluate and args.linear:
        if args.weights is None:
            print('ERROR: --weights required for evaluation')
            return
        if args.model == 'simclr':
            evaluate_simclr(args, device)
        elif args.model == 'dino':
            evaluate_dino(args, device)
        elif args.model == 'mae':
            evaluate_mae(args, device)

    if args.visualize:
        if args.model == 'dino':
            visualize_dino(args, device)
        elif args.model == 'mae':
            visualize_mae(args, device)
        elif args.model == 'tsne':
            visualize_tsne(args, device)


if __name__ == '__main__':
    main()