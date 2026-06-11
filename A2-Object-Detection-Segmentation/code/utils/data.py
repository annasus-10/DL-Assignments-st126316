import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as transforms


# ── Oxford-IIIT Pet Dataset ───────────────────────────────────────────────────

class PetSegDataset(Dataset):
    def __init__(self, base, size=128):
        self.ds = base
        self.img_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((size, size),
                interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor(),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, mask = self.ds[idx]
        img  = self.img_tf(img)
        mask = (self.mask_tf(mask).squeeze(0).long() - 1).clamp(0, 2)
        return img, mask


def get_pet_dataloaders(data_dir='data', img_size=128, batch_size=16):
    train_raw = OxfordIIITPet(data_dir, split='trainval',
                               target_types='segmentation', download=False)
    test_raw  = OxfordIIITPet(data_dir, split='test',
                               target_types='segmentation', download=False)

    train_data   = PetSegDataset(train_raw, img_size)
    test_data    = PetSegDataset(test_raw,  img_size)
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_data,  batch_size=batch_size,
                              shuffle=False, num_workers=2)

    return train_loader, test_loader
    
import json
import albumentations as A
from torchvision.datasets import CocoDetection
from torch.utils.data import Subset
from typing import Any, Callable, Optional, Tuple
from PIL import Image
from utils.loss import iou_xywh_numpy

# YOLOv3 paper anchors
ANCHORS = [
    [[10, 13], [16, 30],  [33, 23]],    # small  (52x52)
    [[30, 61], [62, 45],  [59, 119]],   # medium (26x26)
    [[116, 90],[156, 198],[373, 326]],   # large  (13x13)
]
STRIDES     = [8, 16, 32]
NUM_ANCHORS = 3
NUM_CLASSES = 80


class CustomCoco(CocoDetection):
    def __init__(self, root, annFile, img_size=416,
                 transform=None, target_transform=None, transforms=None):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco     = COCO(annFile)
        self.ids      = list(sorted(self.coco.imgs.keys()))
        self.img_size = img_size

        with open(annFile) as js:
            data = json.load(js)["categories"]
        self.cats_dict = {str(cat['id']): i for i, cat in enumerate(data[:80])}

    def __getitem__(self, index):
        coco   = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target  = coco.loadAnns(ann_ids)
        path    = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = np.array(img)

        category_ids = [obj['category_id'] for obj in target]
        bboxes       = [obj['bbox'] for obj in target]

        if self.transform is not None:
            transformed   = self.transform(image=img, bboxes=bboxes, category_ids=category_ids)
            img           = transformed['image']
            bboxes        = torch.Tensor(transformed['bboxes'])
            cat_ids       = torch.Tensor(transformed['category_ids'])
            labels, bboxes = self.__create_label(bboxes, cat_ids.int(), self.img_size, self.cats_dict)

        return img, labels, bboxes

    def __len__(self):
        return len(self.ids)

    def __create_label(self, bboxes, class_inds, img_size, cats_dict):
        bboxes      = np.array(bboxes)
        class_inds  = np.array(class_inds)
        strides     = np.array(STRIDES)
        train_output_size = img_size / strides

        label = [
            np.zeros((int(train_output_size[i]), int(train_output_size[i]),
                      NUM_ANCHORS, 5 + NUM_CLASSES))
            for i in range(3)
        ]
        bboxes_xywh = [np.zeros((150, 4)) for _ in range(3)]
        bbox_count  = np.zeros((3,))

        for i in range(len(bboxes)):
            bbox_coor      = bboxes[i][:4]
            bbox_class_ind = cats_dict[str(class_inds[i])]

            one_hot = np.zeros(NUM_CLASSES, dtype=np.float32)
            one_hot[bbox_class_ind] = 1.0

            bbox_xywh = np.concatenate(
                [(0.5 * bbox_coor[2:] + bbox_coor[:2]), bbox_coor[2:]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]

            iou          = []
            exist_positive = False
            for j in range(3):
                anchors_xywh = np.zeros((NUM_ANCHORS, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[j, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = ANCHORS[j]

                iou_scale = iou_xywh_numpy(bbox_xywh_scaled[j][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[j, 0:2]).astype(np.int32)
                    label[j][yind, xind, iou_mask, 0:4] = bbox_xywh * strides[j]
                    label[j][yind, xind, iou_mask, 4:5] = 1.0
                    label[j][yind, xind, iou_mask, 5:]  = one_hot
                    bbox_ind = int(bbox_count[j] % 150)
                    bboxes_xywh[j][bbox_ind, :4] = bbox_xywh * strides[j]
                    bbox_count[j] += 1
                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect     = int(best_anchor_ind / NUM_ANCHORS)
                best_anchor     = int(best_anchor_ind % NUM_ANCHORS)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh * strides[best_detect]
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:]  = one_hot
                bbox_ind = int(bbox_count[best_detect] % 150)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh * strides[best_detect]
                bbox_count[best_detect] += 1

        flatten_s = int(train_output_size[2]) ** 2 * NUM_ANCHORS
        flatten_m = int(train_output_size[1]) ** 2 * NUM_ANCHORS
        flatten_l = int(train_output_size[0]) ** 2 * NUM_ANCHORS

        label_s = torch.Tensor(label[2]).view(1, flatten_s, 5 + NUM_CLASSES).squeeze(0)
        label_m = torch.Tensor(label[1]).view(1, flatten_m, 5 + NUM_CLASSES).squeeze(0)
        label_l = torch.Tensor(label[0]).view(1, flatten_l, 5 + NUM_CLASSES).squeeze(0)

        bboxes_s = torch.Tensor(bboxes_xywh[2])
        bboxes_m = torch.Tensor(bboxes_xywh[1])
        bboxes_l = torch.Tensor(bboxes_xywh[0])

        labels  = torch.cat([label_l, label_m, label_s], 0)
        bboxes  = torch.cat([bboxes_l, bboxes_m, bboxes_s], 0)
        return labels, bboxes


def get_coco_dataloaders(path2data, path2json, img_size=416,
                         batch_size=64, total_samples=5000, val_samples=1000):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

    def collate_fn(batch):
        return tuple(zip(*batch))

    full_dataset = CustomCoco(root=path2data, annFile=path2json,
                              img_size=img_size, transform=train_transform)

    train_indices = list(range(0, total_samples - val_samples))
    val_indices   = list(range(total_samples - val_samples, total_samples))

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset   = Subset(full_dataset, val_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, collate_fn=collate_fn)

    print(f'Train: {len(train_dataset)} samples ({len(train_loader)} batches)')
    print(f'Val  : {len(val_dataset)} samples ({len(val_loader)} batches)')
    return train_loader, val_loader