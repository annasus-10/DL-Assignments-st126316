"""Training / evaluation orchestration, dispatched from run.py."""

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ════════════════════════════════════════════
#  Part 1 — Vanilla GAN
# ════════════════════════════════════════════

def train_gan(args, device):
    from models.gan import Generator, Discriminator
    from utils.data import get_mnist_loader
    from utils.viz import save_grid, plot_losses

    Z_DIM = 100
    batch_size = args.batch_size or 128
    loader = get_mnist_loader(args.data_root, batch_size=batch_size)

    G = Generator(Z_DIM).to(device)
    D = Discriminator().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, Z_DIM, device=device)

    tag       = f"_{args.tag}" if args.tag else ""
    ckpt_path = os.path.join(args.save_dir, f"gan_mnist{tag}.pt")
    fig_dir   = os.path.join(args.out_dir, "figures")

    g_losses, d_losses = [], []
    print(f"[GAN] epochs={args.epochs}  d-lr={args.d_lr}  tag='{args.tag}'")

    for epoch in range(args.epochs):
        t0 = time.time()
        g_ep, d_ep = [], []
        for real_imgs, _ in tqdm(loader, desc=f"GAN {epoch+1}/{args.epochs}", leave=False):
            B = real_imgs.size(0)
            real_imgs = real_imgs.view(B, -1).to(device)
            real_lbl  = torch.ones(B, 1, device=device)
            fake_lbl  = torch.zeros(B, 1, device=device)

            z = torch.randn(B, Z_DIM, device=device)
            d_loss = criterion(D(real_imgs), real_lbl) + criterion(D(G(z).detach()), fake_lbl)
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            z = torch.randn(B, Z_DIM, device=device)
            g_loss = criterion(D(G(z)), real_lbl)
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()

            g_ep.append(g_loss.item()); d_ep.append(d_loss.item())

        g_losses.append(np.mean(g_ep)); d_losses.append(np.mean(d_ep))
        print(f"Epoch {epoch+1:02d} | G: {np.mean(g_ep):.3f} | D: {np.mean(d_ep):.3f} | {time.time()-t0:.1f}s")

    torch.save({"G": G.state_dict(), "D": D.state_dict()}, ckpt_path)
    print(f"Saved -> {ckpt_path}")

    G.eval()
    with torch.no_grad():
        fake = G(fixed_noise).view(-1, 1, 28, 28).cpu()
    save_grid(fake, os.path.join(fig_dir, f"gan_grid{tag}.png"), nrow=8,
              title="GAN Generated MNIST" + (" (collapse)" if args.tag else ""))
    plot_losses(g_losses, d_losses, os.path.join(fig_dir, f"gan_losses{tag}.png"))


def evaluate_gan(args, device):
    from models.gan import Generator
    from utils.classifier import train_or_load_classifier
    from utils.viz import plot_digit_histogram

    if not args.weights:
        raise SystemExit("--evaluate requires --weights")

    Z_DIM = 100
    G = Generator(Z_DIM).to(device)
    G.load_state_dict(torch.load(args.weights, map_location=device)["G"])
    G.eval()

    clf = train_or_load_classifier(
        device,
        save_path=os.path.join(args.save_dir, "mnist_classifier.pt"),
        data_root=args.data_root,
    )

    all_preds = []
    with torch.no_grad():
        for start in range(0, 1000, 256):
            n = min(256, 1000 - start)
            z = torch.randn(n, Z_DIM, device=device)
            imgs = G(z).view(-1, 1, 28, 28)
            all_preds.extend(clf(imgs).argmax(1).cpu().tolist())

    counts = [all_preds.count(d) for d in range(10)]
    print("\n--- Mode Collapse Check ---")
    print(f"{'Digit':<8}", " ".join(f"{d:>5}" for d in range(10)))
    print(f"{'Count':<8}", " ".join(f"{c:>5}" for c in counts))

    tag     = f"_{args.tag}" if args.tag else ""
    fig_dir = os.path.join(args.out_dir, "figures")
    title   = "Mode Collapse Histogram (induced)" if args.tag == "collapse" else "Generated Digit Distribution"
    plot_digit_histogram(counts, os.path.join(fig_dir, f"mode_collapse_histogram{tag}.png"), title)


def generate_gan(args, device):
    from models.gan import Generator
    from utils.viz import save_grid
    if not args.weights:
        raise SystemExit("--generate requires --weights")
    G = Generator(100).to(device)
    G.load_state_dict(torch.load(args.weights, map_location=device)["G"])
    G.eval()
    with torch.no_grad():
        imgs = G(torch.randn(args.n, 100, device=device)).view(-1, 1, 28, 28).cpu()
    save_grid(imgs, os.path.join(args.out_dir, "figures", "gan_samples.png"), nrow=8)


# ════════════════════════════════════════════
#  Part 2 — CycleGAN
# ════════════════════════════════════════════

def _denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1)


def train_cyclegan(args, device):
    from models.cyclegan import CycleGenerator, PatchDiscriminator
    from utils.data import get_celeba_loaders
    from utils.viz import save_grid, plot_losses

    tag     = f"_{args.tag}" if args.tag else ""
    fig_dir = os.path.join(args.out_dir, "figures")

    loader_dark, loader_blonde = get_celeba_loaders(
        args.data_root,
        batch_size=args.batch_size or 16,
        max_per_domain=5000,
    )

    G    = CycleGenerator().to(device)
    Fnet = CycleGenerator().to(device)
    D_X  = PatchDiscriminator().to(device)
    D_Y  = PatchDiscriminator().to(device)

    opt_G = torch.optim.Adam(
        list(G.parameters()) + list(Fnet.parameters()), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(
        list(D_X.parameters()) + list(D_Y.parameters()), lr=2e-4, betas=(0.5, 0.999))

    adv_loss   = nn.MSELoss()
    cyc_loss   = nn.L1Loss()
    LAMBDA_CYC = args.lambda_cyc
    LAMBDA_IDT = 5.0

    g_losses, d_losses = [], []
    print(f"[CycleGAN] epochs={args.epochs}  lambda_cyc={LAMBDA_CYC}  tag='{args.tag}'")

    for epoch in range(args.epochs):
        t0 = time.time()
        g_ep, d_ep = [], []
        dark_iter   = iter(loader_dark)
        blonde_iter = iter(loader_blonde)
        n_batches   = min(len(loader_dark), len(loader_blonde))

        for _ in tqdm(range(n_batches), desc=f"CycleGAN {epoch+1}/{args.epochs}", leave=False):
            real_x, _ = next(dark_iter)
            real_y, _ = next(blonde_iter)
            real_x, real_y = real_x.to(device), real_y.to(device)

            opt_G.zero_grad()
            fake_y  = G(real_x);    fake_x  = Fnet(real_y)
            cycle_x = Fnet(fake_y); cycle_y = G(fake_x)
            idt_x   = Fnet(real_x); idt_y   = G(real_y)

            ps       = D_Y(fake_y).shape
            real_lbl = torch.ones(ps, device=device)
            fake_lbl = torch.zeros(ps, device=device)

            loss_adv = adv_loss(D_Y(fake_y), real_lbl) + adv_loss(D_X(fake_x), real_lbl)
            loss_cyc = cyc_loss(cycle_x, real_x) + cyc_loss(cycle_y, real_y)
            loss_idt = cyc_loss(idt_x, real_x)   + cyc_loss(idt_y, real_y)
            loss_G   = loss_adv + LAMBDA_CYC * loss_cyc + LAMBDA_IDT * loss_idt
            loss_G.backward(); opt_G.step()

            opt_D.zero_grad()
            loss_D = 0.5 * (
                adv_loss(D_X(real_x), real_lbl) + adv_loss(D_X(fake_x.detach()), fake_lbl) +
                adv_loss(D_Y(real_y), real_lbl) + adv_loss(D_Y(fake_y.detach()), fake_lbl)
            )
            loss_D.backward(); opt_D.step()

            g_ep.append(loss_G.item()); d_ep.append(loss_D.item())

        g_losses.append(np.mean(g_ep)); d_losses.append(np.mean(d_ep))
        print(f"Epoch {epoch+1:02d} | G: {np.mean(g_ep):.3f} | D: {np.mean(d_ep):.3f} | {time.time()-t0:.1f}s")

    ckpt_path = os.path.join(args.save_dir, f"cyclegan_celeba{tag}.pt")
    torch.save({"G": G.state_dict(), "F": Fnet.state_dict()}, ckpt_path)
    print(f"Saved -> {ckpt_path}")

    G.eval(); Fnet.eval()
    with torch.no_grad():
        bx, _ = next(iter(torch.utils.data.DataLoader(
            torch.utils.data.Subset(loader_dark.dataset,   list(range(4))), batch_size=4)))
        by, _ = next(iter(torch.utils.data.DataLoader(
            torch.utils.data.Subset(loader_blonde.dataset, list(range(4))), batch_size=4)))
        bx, by = bx.to(device), by.to(device)
        fy = G(bx).cpu(); fx = Fnet(by).cpu()

    rows = torch.cat([_denorm(bx.cpu()), _denorm(fy),
                      _denorm(by.cpu()), _denorm(fx)], dim=0)
    save_grid(rows, os.path.join(fig_dir, f"cyclegan_grid{tag}.png"), nrow=4,
              title="Real dark | →Blonde | Real blonde | →Dark")
    plot_losses(g_losses, d_losses, os.path.join(fig_dir, f"cyclegan_losses{tag}.png"),
                title="CycleGAN Training Losses")


def test_cyclegan_face(args, device):
    from models.cyclegan import CycleGenerator
    from utils.viz import save_grid
    import torchvision.transforms as transforms
    from PIL import Image

    if not args.weights or not args.test_image:
        raise SystemExit("Requires --weights and --test-image")
    if not os.path.exists(args.test_image):
        raise SystemExit(f"Image not found: {args.test_image}")

    G    = CycleGenerator().to(device)
    Fnet = CycleGenerator().to(device)
    ckpt = torch.load(args.weights, map_location=device)
    G.load_state_dict(ckpt["G"]); Fnet.load_state_dict(ckpt["F"])
    G.eval(); Fnet.eval()

    img = Image.open(args.test_image).convert("RGB")
    tf  = transforms.Compose([
        transforms.CenterCrop(min(img.size)),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        to_blonde = G(x).squeeze(0).cpu()
        to_dark   = Fnet(x).squeeze(0).cpu()

    imgs = torch.stack([_denorm(x.squeeze(0).cpu()), _denorm(to_blonde), _denorm(to_dark)])
    save_grid(imgs, os.path.join(args.out_dir, "figures", "my_face_result.png"),
              nrow=3, title="Original | → Blonde | → Dark")


# ════════════════════════════════════════════
#  Part 3 — DDPM
# ════════════════════════════════════════════

def _build_ddpm_schedules(betas, device):
    betas      = betas.to(device)
    alphas     = 1.0 - betas
    alpha_bar  = torch.cumprod(alphas, dim=0)
    sqrt_ab    = torch.sqrt(alpha_bar)
    sqrt_1mab  = torch.sqrt(1.0 - alpha_bar)
    sqrt_recip = torch.sqrt(1.0 / alphas)
    prev_ab    = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
    post_var   = betas * (1.0 - prev_ab) / (1.0 - alpha_bar)
    return dict(betas=betas, alpha_bar=alpha_bar, sqrt_ab=sqrt_ab,
                sqrt_1mab=sqrt_1mab, sqrt_recip=sqrt_recip, post_var=post_var)


def train_ddpm(args, device):
    from models.ddpm import SimpleUNet, get_schedule
    from utils.data import get_mnist_loader
    from utils.viz import save_grid, plot_losses, plot_alpha_bar

    T          = 1000
    batch_size = args.batch_size or 128
    loader     = get_mnist_loader(args.data_root, batch_size=batch_size)
    tag        = f"_{args.schedule}" if args.schedule != "linear" else ""
    fig_dir    = os.path.join(args.out_dir, "figures")

    betas = get_schedule(args.schedule, T)
    sc    = _build_ddpm_schedules(betas, device)

    unet   = SimpleUNet().to(device)
    opt    = torch.optim.Adam(unet.parameters(), lr=2e-4)
    losses = []

    print(f"[DDPM] epochs={args.epochs}  schedule={args.schedule}")

    for epoch in range(args.epochs):
        t0 = time.time(); ep_loss = []
        for x0, _ in tqdm(loader, desc=f"DDPM {epoch+1}/{args.epochs}", leave=False):
            x0  = x0.to(device)
            B   = x0.size(0)
            t   = torch.randint(0, T, (B,), device=device)
            eps = torch.randn_like(x0)
            x_t = sc["sqrt_ab"][t][:, None, None, None] * x0 + \
                  sc["sqrt_1mab"][t][:, None, None, None] * eps
            pred = unet(x_t, t)
            loss = F.mse_loss(pred, eps)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss.append(loss.item())

        losses.append(np.mean(ep_loss))
        print(f"Epoch {epoch+1:03d} | Loss: {np.mean(ep_loss):.4f} | {time.time()-t0:.1f}s")

    ckpt_path = os.path.join(args.save_dir, f"ddpm_mnist{tag}.pt")
    torch.save({"unet": unet.state_dict(), "schedule": args.schedule}, ckpt_path)
    print(f"Saved -> {ckpt_path}")

    from models.ddpm import linear_beta_schedule, cosine_beta_schedule
    ab_lin = torch.cumprod(1.0 - linear_beta_schedule(T), dim=0)
    ab_cos = torch.cumprod(1.0 - cosine_beta_schedule(T), dim=0)
    plot_alpha_bar(ab_lin, ab_cos, os.path.join(fig_dir, "schedule_comparison.png"))
    plot_losses(losses, losses, os.path.join(fig_dir, f"ddpm_losses{tag}.png"),
                title=f"DDPM Training Loss ({args.schedule})")


def generate_ddpm(args, device):
    from models.ddpm import SimpleUNet, get_schedule
    from utils.viz import save_grid

    if not args.weights:
        raise SystemExit("--generate requires --weights")

    T        = 1000
    ckpt     = torch.load(args.weights, map_location=device)
    schedule = ckpt.get("schedule", "linear")
    betas    = get_schedule(schedule, T)
    sc       = _build_ddpm_schedules(betas, device)

    unet = SimpleUNet().to(device)
    unet.load_state_dict(ckpt["unet"])
    unet.eval()

    tag     = f"_{schedule}"
    fig_dir = os.path.join(args.out_dir, "figures")

    x = torch.randn(args.n, 1, 28, 28, device=device)
    with torch.no_grad():
        for t in tqdm(reversed(range(T)), total=T, desc="Sampling"):
            t_b  = torch.full((args.n,), t, device=device, dtype=torch.long)
            pred = unet(x, t_b)
            coef = sc["betas"][t] / sc["sqrt_1mab"][t]
            mean = sc["sqrt_recip"][t] * (x - coef * pred)
            x    = mean if t == 0 else mean + torch.sqrt(sc["post_var"][t]) * torch.randn_like(x)

    save_grid(x.cpu(), os.path.join(fig_dir, f"ddpm_grid{tag}.png"), nrow=8,
              title=f"DDPM Samples ({schedule} schedule)")

    # Denoising trajectory
    x = torch.randn(8, 1, 28, 28, device=device)
    snapshots = []
    show_at = {999, 800, 600, 400, 200, 100, 50, 0}
    with torch.no_grad():
        for t in reversed(range(T)):
            t_b  = torch.full((8,), t, device=device, dtype=torch.long)
            pred = unet(x, t_b)
            coef = sc["betas"][t] / sc["sqrt_1mab"][t]
            mean = sc["sqrt_recip"][t] * (x - coef * pred)
            x    = mean if t == 0 else mean + torch.sqrt(sc["post_var"][t]) * torch.randn_like(x)
            if t in show_at:
                snapshots.append((t, x.cpu().clone()))

    import matplotlib.pyplot as plt
    snapshots.sort(key=lambda s: s[0], reverse=True)
    fig, axes = plt.subplots(8, len(snapshots), figsize=(len(snapshots) * 1.5, 12))
    for col, (t_val, imgs) in enumerate(snapshots):
        for row in range(8):
            axes[row][col].imshow(imgs[row].squeeze().numpy(), cmap="gray")
            axes[row][col].axis("off")
            if row == 0:
                axes[row][col].set_title(f"t={t_val}", fontsize=9)
    plt.suptitle("Reverse Diffusion: Noise → Digit", fontsize=13)
    plt.tight_layout()
    traj_path = os.path.join(fig_dir, f"ddpm_trajectory{tag}.png")
    plt.savefig(traj_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved: {traj_path}")