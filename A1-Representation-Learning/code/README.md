# A1: Representation Learning

## Training Commands

```bash
# From scratch
python3 run.py --model alexnet --dataset cifar10 --epochs 10 --batch_size 64 --train
python3 run.py --model alexnet_lrn --dataset cifar10 --epochs 10 --batch_size 64 --train
python3 run.py --model googlenet --dataset cifar10 --epochs 25 --batch_size 64 --train
python3 run.py --model resnet18 --dataset cifar10 --epochs 20 --batch_size 64 --train
python3 run.py --model vit_small --dataset cifar10 --epochs 20 --batch_size 64 --train

# Pretrained fine-tuning
python3 run.py --model alexnet_pretrained --dataset cifar10 --epochs 15 --batch_size 64 --train
python3 run.py --model googlenet_pretrained --dataset cifar10 --epochs 15 --batch_size 64 --train
python3 run.py --model resnet18_pretrained --dataset cifar10 --epochs 15 --batch_size 64 --train
python3 run.py --model vit_b16_pretrained --dataset cifar10 --epochs 15 --batch_size 64 --train
```

## Results

| Model | # Params | Val Accuracy | Time/epoch | Architecture Type |
|---|---|---|---|---|
| AlexNet (from scratch) | 58,322,314 | 53.29% | ~50s | CNN |
| AlexNet + LRN (from scratch) | 58,322,314 | 64.00% | ~50s | CNN |
| GoogLeNet (from scratch) | 10,334,030 | ? | ~105s | CNN + Inception |
| ResNet-18 (from scratch) | 11,181,642 | 82.08% | ~65s | CNN + Skip connections |
| ViT-Small (from scratch) | 1,205,898 | 35.39% | ~30s | Transformer |
| AlexNet (pretrained) | 57,044,810 | 90.07% | ~50s | CNN |
| GoogLeNet (pretrained) | 5,610,154 | ? | ~93s | CNN + Inception |
| ResNet-18 (pretrained) | ? | ? | ? | CNN + Skip connections |
| ViT-B/16 (pretrained) | ? | ? | ? | Transformer |

## Discussion

ResNet-18 achieved the best accuracy among from-scratch models at 82.08%, demonstrating that skip connections effectively solve the vanishing gradient problem and allow deeper networks to train successfully on CIFAR-10. Pretrained models significantly outperformed their from-scratch counterparts — pretrained AlexNet reached 90.07% compared to 53.29% from scratch, showing the power of ImageNet feature transfer even to a very different dataset.

Adding Local Response Normalization to AlexNet improved accuracy from 53.29% to 64.00%, confirming its role in the original paper as a meaningful regularization technique, though modern networks have replaced it entirely with Batch Normalization.

ViT-Small from scratch performed poorly at 35.39%, which is expected — Transformers lack the inductive biases of CNNs (locality, translation equivariance) and require much more data to learn these patterns. The pretrained ViT-B/16 is expected to reverse this trend entirely, as large-scale pretraining compensates for the lack of inductive bias.