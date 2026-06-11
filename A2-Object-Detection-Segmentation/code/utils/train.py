import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


def compute_map(model, dataloader, device, img_size, conf_thresh=0.05, nms_thresh=0.4, num_classes=80):
    """Compute mAP@0.5 over the dataloader using torchmetrics."""
    try:
        from torchmetrics.detection import MeanAveragePrecision
        from torchvision.ops import nms as tv_nms
    except ImportError:
        print("torchmetrics/torchvision not installed — skipping mAP")
        return None

    warnings.filterwarnings("ignore", message="Encountered more than")
    metric = MeanAveragePrecision(
        iou_type="bbox",
        iou_thresholds=[0.5],
        max_detection_thresholds=[1, 10, 100],
    )
    model.eval()

    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for inputs, labels, bboxes in tqdm(dataloader, desc="  mAP eval", leave=False):
            inputs = torch.from_numpy(np.array(inputs)).squeeze(1).permute(0, 3, 1, 2).float().to(device)
            use_cuda = (device.type == "cuda")
            outputs = model(inputs, use_cuda)

            pred_conf = torch.sigmoid(outputs[..., 4])
            pred_cls  = torch.sigmoid(outputs[..., 5:])
            pred_xy   = outputs[..., 0:2]
            pred_wh   = outputs[..., 2:4]

            pred_x1 = (pred_xy[..., 0] - pred_wh[..., 0] / 2).clamp(0, img_size)
            pred_y1 = (pred_xy[..., 1] - pred_wh[..., 1] / 2).clamp(0, img_size)
            pred_x2 = (pred_xy[..., 0] + pred_wh[..., 0] / 2).clamp(0, img_size)
            pred_y2 = (pred_xy[..., 1] + pred_wh[..., 1] / 2).clamp(0, img_size)
            pred_boxes_xyxy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)

            cls_scores, cls_ids = pred_cls.max(dim=-1)
            scores = pred_conf * cls_scores

            preds, targets_list = [], []
            for i in range(inputs.size(0)):
                mask = scores[i] > conf_thresh
                if mask.sum() == 0:
                    preds.append(dict(
                        boxes=torch.zeros((0, 4)),
                        scores=torch.zeros(0),
                        labels=torch.zeros(0, dtype=torch.long)
                    ))
                else:
                    b = pred_boxes_xyxy[i][mask].cpu()
                    s = scores[i][mask].cpu()
                    l = cls_ids[i][mask].cpu().long()
                    keep = tv_nms(b, s, nms_thresh)
                    keep = keep[:100]
                    preds.append(dict(boxes=b[keep], scores=s[keep], labels=l[keep]))

                label_i  = labels[i] if isinstance(labels, (list, tuple)) else torch.stack(labels)[i]
                obj_mask = label_i[..., 4] > 0
                gt_xywh  = label_i[obj_mask][..., :4]
                gt_cls_oh = label_i[obj_mask][..., 5:]
                if gt_xywh.shape[0] == 0:
                    targets_list.append(dict(
                        boxes=torch.zeros((0, 4)),
                        labels=torch.zeros(0, dtype=torch.long)
                    ))
                    continue
                gt_cls_ids = gt_cls_oh.argmax(dim=-1).long()
                gt_x1 = (gt_xywh[..., 0] - gt_xywh[..., 2] / 2).clamp(0)
                gt_y1 = (gt_xywh[..., 1] - gt_xywh[..., 3] / 2).clamp(0)
                gt_x2 = (gt_xywh[..., 0] + gt_xywh[..., 2] / 2)
                gt_y2 = (gt_xywh[..., 1] + gt_xywh[..., 3] / 2)
                targets_list.append(dict(
                    boxes=torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=-1).cpu(),
                    labels=gt_cls_ids.cpu(),
                ))

            metric.update(preds, targets_list)

    try:
        result = metric.compute()
        return result['map_50'].item()
    except Exception:
        return 0.0


def run_training(model, optimizer, dataloader, val_dataloader, device, img_size,
                 n_epoch, every_n_batch, every_n_epoch, ckpt_dir, use_ciou=False):
    os.makedirs(ckpt_dir, exist_ok=True)
    history = {"loss": [], "box": [], "conf": [], "cls": [], "map50": []}
    print(f"Starting training for {n_epoch} epochs...")

    for epoch_i in range(n_epoch):
        t0 = time.time()
        running_loss = running_ciou = running_conf = running_cls = 0.0
        n_batches = 0
        n_skipped = 0

        model.train()
        pbar = tqdm(dataloader, desc=f'Epoch {epoch_i+1}/{n_epoch}')
        for inputs, labels, bboxes in pbar:
            inputs = torch.from_numpy(np.array(inputs)).squeeze(1).permute(0, 3, 1, 2).float().to(device)
            labels = torch.stack(labels).to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs, True)

                pred_xywh = outputs[..., 0:4] / img_size
                raw_conf  = outputs[..., 4:5]
                raw_cls   = outputs[..., 5:]

                label_xywh       = labels[..., :4] / img_size
                label_obj_mask   = labels[..., 4:5].clamp(0, 1)
                label_noobj_mask = 1.0 - label_obj_mask
                label_cls        = labels[..., 5:].clamp(0, 1)

                lambda_coord = 1.0
                lambda_noobj = 0.5

                bce_logits = nn.BCEWithLogitsLoss(reduction='none')
                mse        = nn.MSELoss(reduction='none')
                batch_norm = inputs.size(0)

                if use_ciou:
                    from utils.loss import CIOU_xywh_torch
                    pred_flat  = pred_xywh.view(-1, 4)
                    label_flat = label_xywh.view(-1, 4)
                    mask_flat  = label_obj_mask.view(-1)
                    ciou_vals  = CIOU_xywh_torch(pred_flat, label_flat)
                    loss_box   = lambda_coord * torch.sum(mask_flat * (1 - ciou_vals)) / batch_norm
                else:
                    loss_xy  = lambda_coord * torch.sum(label_obj_mask * mse(pred_xywh[..., :2], label_xywh[..., :2])) / batch_norm
                    loss_wh  = lambda_coord * torch.sum(label_obj_mask * mse(pred_xywh[..., 2:], label_xywh[..., 2:])) / batch_norm
                    loss_box = loss_xy + loss_wh

                loss_conf = (torch.sum(label_obj_mask   * bce_logits(raw_conf, label_obj_mask)) +
                             lambda_noobj * torch.sum(label_noobj_mask * bce_logits(raw_conf, label_obj_mask))) / batch_norm
                loss_cls  = torch.sum(label_obj_mask * bce_logits(raw_cls, label_cls)) / batch_norm
                loss      = loss_box + loss_conf + loss_cls

                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    n_skipped += 1
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

                n = inputs.size(0)
                running_loss += loss.item()     * n
                running_ciou += loss_box.item() * n
                running_conf += loss_conf.item()* n
                running_cls  += loss_cls.item() * n
                n_batches    += 1

                avg_loss = running_loss / (n_batches * n)
                avg_ciou = running_ciou / (n_batches * n)
                avg_conf = running_conf / (n_batches * n)
                pbar.set_postfix(loss=f'{avg_loss:.2f}', ciou=f'{avg_ciou:.2f}', conf=f'{avg_conf:.2f}')

        if n_skipped > 0:
            print(f"  [warn] skipped {n_skipped}/{n_batches+n_skipped} batches due to nan loss")

        denom      = max(n_batches * 5, 1)
        epoch_loss = running_loss / denom
        epoch_ciou = running_ciou / denom
        epoch_conf = running_conf / denom
        epoch_cls  = running_cls  / denom
        elapsed    = time.time() - t0

        map50   = compute_map(model, val_dataloader, device, img_size)
        map_str = f"{map50:.4f}" if map50 is not None else "  N/A  "

        history["loss"].append(epoch_loss)
        history["box"].append(epoch_ciou)
        history["conf"].append(epoch_conf)
        history["cls"].append(epoch_cls)
        history["map50"].append(map50 if map50 is not None else 0.0)

        print(f"Epoch {epoch_i+1:02d}/{n_epoch} | Loss: {epoch_loss:.4f} | Box: {epoch_ciou:.4f} | "
              f"Conf: {epoch_conf:.4f} | Cls: {epoch_cls:.4f} | mAP@50(val): {map_str} | Time: {elapsed:.1f}s")

        if every_n_epoch:
            ckpt_path = os.path.join(ckpt_dir, f"yolov4_epoch{epoch_i+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"         └─ saved → {ckpt_path}")

    print("Training complete.")
    return history


# ── Segmentation Training ─────────────────────────────────────────────────────

def train_segmentation(model, train_loader, test_loader, optimizer, scheduler,
                       device, epochs, save_name):
    from utils.eval import compute_iou
    criterion = nn.CrossEntropyLoss()
    train_losses, val_ious = [], []
    best_iou = 0.0

    for epoch in range(epochs):
        model.train()
        ep_loss = []

        for imgs, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            imgs, masks = imgs.to(device), masks.to(device)
            loss = criterion(model(imgs), masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss.append(loss.item())

        model.eval()
        ep_iou = []
        with torch.no_grad():
            for imgs, masks in test_loader:
                ep_iou.append(compute_iou(model(imgs.to(device)), masks.to(device)))

        scheduler.step()
        train_losses.append(np.mean(ep_loss))
        val_ious.append(np.mean(ep_iou))
        print(f'Epoch {epoch+1:02d} | Loss: {train_losses[-1]:.4f} | mIoU: {val_ious[-1]:.4f}')

        if val_ious[-1] > best_iou:
            best_iou = val_ious[-1]
            torch.save(model.state_dict(), save_name)
            print(f'  → Saved best checkpoint: {save_name}')

    print(f'Best mIoU: {best_iou:.4f}')
    return train_losses, val_ious