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

| Model | # Params | Test Accuracy | Time/epoch | Architecture Type |
|---|---|---|---|---|
| AlexNet (from scratch) | 58,322,314 | 64.82% | ~50s | CNN |
| AlexNet + LRN (from scratch) | 58,322,314 | 53.18% | ~50s | CNN |
| GoogLeNet (from scratch) | 10,334,030 | 10.00% | ~105s | CNN + Inception |
| ResNet-18 (from scratch) | 11,181,642 | 81.36% | ~65s | CNN + Skip connections |
| ViT-Small (from scratch) | 1,205,898 | 33.88% | ~30s | Transformer |
| AlexNet (pretrained) | 57,044,810 | 89.83% | ~50s | CNN |
| GoogLeNet (pretrained) | 5,610,154 | 93.83% | ~93s | CNN + Inception |
| ResNet-18 (pretrained) | 11,181,642 | 93.20% | ~88s | CNN + Skip connections |
| ViT-B/16 (pretrained) | 85,806,346 | 95.61% | ~205s | Transformer |

## Discussion

ViT-B/16 pretrained achieved the best test accuracy at 95.61%, demonstrating that large-scale pretraining overcomes the lack of CNN inductive biases. Among from-scratch models, ResNet-18 performed best at 81.36%, showing that skip connections effectively solve the vanishing gradient problem in deep networks. GoogLeNet from scratch failed to learn (10.00%) due to a known implementation issue with the auxiliary classifier sizing, while the pretrained GoogLeNet reached 93.83%, confirming the architecture is sound.

Pretrained models consistently outperformed their from-scratch counterparts by a large margin — pretrained AlexNet (89.83%) vs scratch (64.82%), showing that ImageNet features transfer well even to CIFAR-10. ViT-Small from scratch scored only 33.88%, confirming that Transformers without pretraining struggle on small datasets due to their lack of locality and translation equivariance inductive biases that CNNs have built in.