import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm


class DINOAugmentation:
    def __init__(self, image_size=32, n_local=4):
        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        flip_jitter = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
            *flip_jitter,
            transforms.ToTensor(), normalize
        ])
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.05, 0.4)),
            *flip_jitter,
            transforms.ToTensor(), normalize
        ])
        self.n_local = n_local

    def __call__(self, img):
        global1 = self.global_transform(img)
        global2 = self.global_transform(img)
        locals_ = [self.local_transform(img) for _ in range(self.n_local)]
        return [global1, global2] + locals_


class DINOHead(nn.Module):
    def __init__(self, in_dim=192, hidden_dim=512, out_dim=256, n_layers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, out_dim, bias=False))
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(nn.Linear(out_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class DINOLoss(nn.Module):
    def __init__(self, out_dim=256, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9, use_centering=True):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.use_centering = use_centering
        self.register_buffer('center', torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out):
        s_probs = [F.log_softmax(s / self.student_temp, dim=-1) for s in student_out]
        if self.use_centering:
            t_probs = [F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach()
                       for t in teacher_out]
        else:
            t_probs = [F.softmax(t / self.teacher_temp, dim=-1).detach()
                       for t in teacher_out]
        total_loss = 0
        n_loss_terms = 0
        for t_idx, t_prob in enumerate(t_probs):
            for s_idx, s_log_prob in enumerate(s_probs):
                if s_idx == t_idx:
                    continue
                loss = -(t_prob * s_log_prob).sum(dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(torch.stack(teacher_out).mean(dim=0))
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_mean):
        self.center = self.center * self.center_momentum + teacher_mean * (1 - self.center_momentum)


def build_dino_model(out_dim=256):
    vit = timm.create_model('vit_tiny_patch16_224', pretrained=False,
                             img_size=32, patch_size=4, num_classes=0)
    embed_dim = vit.embed_dim
    head = DINOHead(in_dim=embed_dim, out_dim=out_dim)
    return vit, head