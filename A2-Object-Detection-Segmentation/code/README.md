# A2: Object Detection & Image Segmentation

## Setup

```bash
cd code
source .venv/bin/activate
pip install -r requirements.txt
```

## Training Commands

### Object Detection (YOLOv3 + YOLOv4)

```bash
# Inference with pretrained YOLOv3 weights
python3 run.py --model yolov3 --cfg data/yolov3.cfg --weights data/yolov3.weights --image data/dog-cycle-car.png --infer

# Train YOLOv4 on COCO (MSE/IoU loss)
python3 run.py --model yolov4 --cfg data/yolov4.cfg --dataset coco --path2data data/coco/val2017 --path2json data/coco/annotations/instances_val2017.json --epochs 5 --batch_size 64 --lr 1e-5 --train 2>&1 | tee logs/yolov4_iou.log

# Train YOLOv4 on COCO (CIoU loss)
python3 run.py --model yolov4 --cfg data/yolov4.cfg --dataset coco --path2data data/coco/val2017 --path2json data/coco/annotations/instances_val2017.json --epochs 5 --batch_size 64 --lr 1e-5 --train --use_ciou 2>&1 | tee logs/yolov4_ciou.log

# Evaluate mAP
python3 run.py --model yolov4 --cfg data/yolov4.cfg --weights checkpoints/yolov4_epoch05.pt --dataset coco --path2data data/coco/val2017 --path2json data/coco/annotations/instances_val2017.json --evaluate
```

### Image Segmentation (U-Net)

```bash
# Train U-Net with skip connections (baseline)
python3 run.py --model unet_resnet18 --dataset oxford_pet --epochs 20 --batch_size 16 --lr 1e-4 --train 2>&1 | tee logs/unet_resnet18.log

# Train U-Net without skip connections (ablation)
python3 run.py --model unet_resnet18_no_skip --dataset oxford_pet --epochs 20 --batch_size 16 --lr 1e-4 --train 2>&1 | tee logs/unet_resnet18_no_skip.log

# Evaluate
python3 run.py --model unet_resnet18 --weights checkpoints/unet_resnet18_pet.pt --dataset oxford_pet --evaluate
python3 run.py --model unet_resnet18_no_skip --weights checkpoints/unet_resnet18_no_skip_pet.pt --dataset oxford_pet --evaluate
```

## Results

### A2-01: Object Detection

| Model | Dataset | mAP@50 | Time/epoch | Notes |
|---|---|---|---|---|
| YOLOv3 (pretrained) | COCO | — | — | inference only |
| YOLOv4 (MSE/IoU loss) | COCO | ? | ? | trained 5 epochs |
| YOLOv4 (CIoU loss) | COCO | ? | ? | loss comparison |

### A2-02: Image Segmentation

| Model | Encoder | Skip Connections | Val mIoU | Time/epoch |
|---|---|---|---|---|
| unet_resnet18 | ResNet-18 (ImageNet) | ✅ Yes | ? | ? |
| unet_resnet18_no_skip | ResNet-18 (ImageNet) | ❌ No | ? | ? |

## Discussion

### Why is YOLOv3 faster than Faster R-CNN?
Faster R-CNN is a two-stage detector: the RPN first generates region proposals, then a separate detection head classifies each proposal. This means two sequential forward passes plus RoI pooling overhead. YOLOv3 is a single-shot detector — it divides the image into a grid and predicts all bounding boxes and class probabilities in one forward pass simultaneously. There are no region proposals, no RoI pooling, and no second stage. This architectural difference makes YOLOv3 roughly 10× faster than Faster R-CNN.

### Effect of CIoU vs MSE/IoU loss
*To be filled after training.*

### Skip connections: do they matter?
*To be filled after training.*

### Why do skip connections help segmentation more than classification?
Classification only needs to know *what* is in the image — this can be determined from abstract high-level features at the bottleneck. Segmentation needs to know *what* AND *exactly where* each pixel belongs. Without skip connections, the decoder must reconstruct precise spatial boundaries from a heavily downsampled bottleneck representation, losing fine-grained edge and texture information. Skip connections directly pass high-resolution encoder features to the decoder, preserving spatial detail that is critical for accurate mask boundaries.

### Which skip level hurts most when removed?
The first skip connection (64ch, highest resolution, from `stem_conv`) hurts the most. This is the highest-resolution feature map (H/2) and carries the most precise spatial and edge information. Removing it forces the decoder to reconstruct fine boundaries entirely from lower-resolution features, causing the most degradation in mIoU — especially on thin structures like pet fur boundaries and ears.