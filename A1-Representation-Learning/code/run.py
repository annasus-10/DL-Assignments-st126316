import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import vit_b_16, ViT_B_16_Weights

from models.alexnet import AlexNet, AlexNetLRN
from models.googlenet import GoogLeNet
from models.resnet import ResNet18
from models.vit import ViTSmall
from utils.data import get_dataloaders
from utils.train import train_model, evaluate_model


def get_model(model_name, device):
    if model_name == 'alexnet':
        return AlexNet(), False

    elif model_name == 'alexnet_lrn':
        return AlexNetLRN(), False

    elif model_name == 'alexnet_pretrained':
        model = torchvision.models.alexnet(weights='IMAGENET1K_V1')
        model.classifier[6] = nn.Linear(4096, 10)
        return model, False

    elif model_name == 'googlenet':
        return GoogLeNet(), True   # is_inception=True

    elif model_name == 'googlenet_pretrained':
        model = torchvision.models.googlenet(weights='IMAGENET1K_V1')
        model.aux_logits = False
        model.aux1 = None
        model.aux2 = None
        model.fc = nn.Linear(1024, 10)
        return model, False

    elif model_name == 'resnet18':
        return ResNet18(), False

    elif model_name == 'resnet18_pretrained':
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(512, 10)
        return model, False

    elif model_name == 'vit_small':
        return ViTSmall(), False

    elif model_name == 'vit_b16_pretrained':
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        model.heads = nn.Linear(768, 10)
        return model, False

    else:
        raise ValueError(f'Unknown model: {model_name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      required=True)
    parser.add_argument('--dataset',    default='cifar10')
    parser.add_argument('--epochs',     type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr',         type=float, default=0.001)
    parser.add_argument('--train',      action='store_true')
    parser.add_argument('--test',       action='store_true')
    parser.add_argument('--weights',    type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ViTSmall needs 32x32, everything else 224x224
    img_size = 32 if args.model == 'vit_small' else 224
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, img_size=img_size)

    model, is_inception = get_model(args.model, device)
    model = model.to(device)

    print(f'Model: {args.model}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # ------------------------------------------------------------------ #
    #  TRAIN                                                             #
    # ------------------------------------------------------------------ #
    if args.train:
        criterion = nn.CrossEntropyLoss()

        # Pretrained two-stage fine-tuning
        if args.model in ('resnet18_pretrained', 'vit_b16_pretrained',
                          'alexnet_pretrained', 'googlenet_pretrained'):

            print('\n--- Stage 1: training head only (5 epochs) ---')
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze the right head for each model
            if args.model == 'resnet18_pretrained':
                model.fc.requires_grad_(True)
                head_params = model.fc.parameters()
            elif args.model == 'vit_b16_pretrained':
                model.heads.requires_grad_(True)
                head_params = model.heads.parameters()
            elif args.model == 'alexnet_pretrained':
                model.classifier[6].requires_grad_(True)
                head_params = model.classifier[6].parameters()
            elif args.model == 'googlenet_pretrained':
                model.fc.requires_grad_(True)
                head_params = model.fc.parameters()

            optimizer = optim.Adam(head_params, lr=1e-3)
            dataloaders = {'train': train_loader, 'val': val_loader}
            train_model(model, dataloaders, criterion, optimizer, device,
                        num_epochs=5, weights_name=f'{args.model}_cifar10',
                        is_inception=is_inception)

            print('\n--- Stage 2: fine-tuning all layers ---')
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            train_model(model, dataloaders, criterion, optimizer, device,
                        num_epochs=args.epochs - 5,
                        weights_name=f'{args.model}_cifar10',
                        is_inception=is_inception)

        # From-scratch training
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
            dataloaders = {'train': train_loader, 'val': val_loader}
            train_model(model, dataloaders, criterion, optimizer, device,
                        num_epochs=args.epochs,
                        weights_name=f'{args.model}_cifar10',
                        is_inception=is_inception)

    # ------------------------------------------------------------------ #
    #  TEST                                                              #
    # ------------------------------------------------------------------ #
    if args.test:
        if args.weights is None:
            args.weights = f'{args.model}_cifar10.pth'
        print(f'Loading weights from {args.weights}')
        model.load_state_dict(torch.load(args.weights, map_location=device))
        evaluate_model(model, test_loader, device)


if __name__ == '__main__':
    main()