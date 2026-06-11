import torch
import numpy as np


def compute_iou(pred, target, n_classes=3):
    """Compute mean IoU over a batch. pred: (N, C, H, W) logits, target: (N, H, W)."""
    pred = pred.argmax(dim=1)
    ious = []
    for cls in range(n_classes):
        inter = ((pred == cls) & (target == cls)).sum().float()
        union = ((pred == cls) | (target == cls)).sum().float()
        if union > 0:
            ious.append((inter / union).item())
    return np.mean(ious) if ious else 0.0


def evaluate_segmentation(model, loader, device, n_classes=3):
    """Run evaluation over full dataloader, return mean mIoU."""
    model.eval()
    all_ious = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            all_ious.append(compute_iou(outputs, masks, n_classes))
    return np.mean(all_ious)


def evaluate_detection(model, loader, device, conf_thresh=0.5, nms_thresh=0.4):
    """Compute mAP@50 for YOLO model over full dataloader."""
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    from utils.util import write_results

    metric = MeanAveragePrecision(iou_type='bbox')
    model.eval()
    CUDA = device.type == 'cuda'

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            outputs = model(imgs, CUDA)

            # NMS
            results = write_results(outputs, conf_thresh, 80, nms_conf=nms_thresh)

            preds_list = []
            tgts_list  = []

            for b in range(imgs.shape[0]):
                if isinstance(results, int):
                    preds_list.append({
                        'boxes':  torch.zeros((0, 4), device='cpu'),
                        'scores': torch.zeros(0, device='cpu'),
                        'labels': torch.zeros(0, dtype=torch.long, device='cpu'),
                    })
                else:
                    det = results[results[:, 0] == b]
                    preds_list.append({
                        'boxes':  det[:, 1:5].cpu(),
                        'scores': det[:, 5].cpu(),
                        'labels': det[:, 7].long().cpu(),
                    })

                bt = targets[targets[:, 0] == b]
                tgts_list.append({
                    'boxes':  bt[:, 2:6].cpu(),
                    'labels': bt[:, 1].long().cpu(),
                })

            metric.update(preds_list, tgts_list)

    result = metric.compute()
    return result['map_50'].item()