import argparse
import os
import torch
import torch.optim as optim

from models.unet import UNet
from models.unet_resnet18 import UNetResNet18, UNetResNet18NoSkip
from models.yolov3 import MyDarknet
from models.yolov4 import MyDarknetV4
from utils.data import get_pet_dataloaders, get_coco_dataloaders
from utils.train import run_training, train_segmentation, compute_map
from utils.eval import evaluate_segmentation


def get_model(args, device):
    if args.model == 'unet':
        return UNet(n_classes=3)

    elif args.model == 'unet_resnet18':
        return UNetResNet18(n_classes=3, pretrained=True)

    elif args.model == 'unet_resnet18_no_skip':
        return UNetResNet18NoSkip(n_classes=3, pretrained=True)

    elif args.model == 'yolov3':
        assert args.cfg, '--cfg required for yolov3'
        model = MyDarknet(args.cfg)
        if args.weights and args.weights.endswith('.weights'):
            model.load_weights(args.weights)
            print(f'Loaded pretrained weights: {args.weights}')
        elif args.weights:
            model.load_state_dict(torch.load(args.weights, map_location=device))
            print(f'Loaded checkpoint: {args.weights}')
        return model

    elif args.model == 'yolov4':
        assert args.cfg, '--cfg required for yolov4'
        model = MyDarknetV4(args.cfg)
        if args.weights and args.weights.endswith('.weights'):
            model.load_weights(args.weights)
            print(f'Loaded pretrained weights: {args.weights}')
        elif args.weights:
            model.load_state_dict(torch.load(args.weights, map_location=device))
            print(f'Loaded checkpoint: {args.weights}')
        return model

    else:
        raise ValueError(f'Unknown model: {args.model}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',        required=True,
                        choices=['unet', 'unet_resnet18', 'unet_resnet18_no_skip',
                                 'yolov3', 'yolov4'])
    parser.add_argument('--dataset',      default=None,
                        choices=['oxford_pet', 'coco'])
    parser.add_argument('--epochs',       type=int,   default=20)
    parser.add_argument('--batch_size',   type=int,   default=16)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--train',        action='store_true')
    parser.add_argument('--evaluate',     action='store_true')
    parser.add_argument('--infer',        action='store_true')
    parser.add_argument('--weights',      type=str,   default=None)
    parser.add_argument('--cfg',          type=str,   default=None)
    parser.add_argument('--image',        type=str,   default=None)
    parser.add_argument('--use_ciou',     action='store_true')
    parser.add_argument('--path2data',    type=str,   default=None)
    parser.add_argument('--path2json',    type=str,   default=None)
    args = parser.parse_args()

    # Infer dataset from model if not specified
    if args.dataset is None:
        args.dataset = 'coco' if args.model in ('yolov3', 'yolov4') else 'oxford_pet'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = get_model(args, device).to(device)
    print(f'Model:      {args.model}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # ------------------------------------------------------------------ #
    #  SEGMENTATION                                                        #
    # ------------------------------------------------------------------ #
    if args.dataset == 'oxford_pet':
        train_loader, test_loader = get_pet_dataloaders(
            data_dir='data', batch_size=args.batch_size)

        if args.train:
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
            save_name = f'checkpoints/{args.model}_pet.pt'
            train_segmentation(model, train_loader, test_loader,
                               optimizer, scheduler, device,
                               args.epochs, save_name)

        if args.evaluate:
            weights = args.weights or f'checkpoints/{args.model}_pet.pt'
            model.load_state_dict(torch.load(weights, map_location=device))
            miou = evaluate_segmentation(model, test_loader, device)
            print(f'Test mIoU: {miou:.4f}')

    # ------------------------------------------------------------------ #
    #  DETECTION                                                           #
    # ------------------------------------------------------------------ #
    elif args.dataset == 'coco':
        if args.train or args.evaluate:
            assert args.path2data, '--path2data required for coco'
            assert args.path2json, '--path2json required for coco'

            img_size = 416

            train_loader, val_loader = get_coco_dataloaders(
                path2data=args.path2data,
                path2json=args.path2json,
                img_size=img_size,
                batch_size=args.batch_size)

            if args.train:
                optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                      momentum=0.9, weight_decay=5e-4)
                run_training(
                    model=model,
                    optimizer=optimizer,
                    dataloader=train_loader,
                    val_dataloader=val_loader,
                    device=device,
                    img_size=img_size,
                    n_epoch=args.epochs,
                    every_n_batch=False,
                    every_n_epoch=True,
                    ckpt_dir='checkpoints',
                    use_ciou=args.use_ciou,
                )

            if args.evaluate:
                map50 = compute_map(model, val_loader, device, img_size)
                print(f'mAP@50: {map50:.4f}')
    # ------------------------------------------------------------------ #
    #  INFERENCE ON SINGLE IMAGE                                           #
    # ------------------------------------------------------------------ #
    if args.infer:
        assert args.image,   '--image required for --infer'
        assert args.weights, '--weights required for --infer'
        import cv2
        import numpy as np
        from utils.util import write_results

        CUDA    = device.type == 'cuda'
        inp_dim = 608 if args.model == 'yolov4' else 416

        img     = cv2.imread(args.image)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (inp_dim, inp_dim))
        img_t   = torch.from_numpy(img_res).permute(2, 0, 1).float().div(255.0)
        img_t   = img_t.unsqueeze(0).to(device)

        with torch.no_grad():
            output  = model(img_t, CUDA)
            results = write_results(output, 0.5, 80, nms_conf=0.4)

        if isinstance(results, int):
            print('No detections.')
        else:
            print(f'Detections: {results.shape[0]}')
            print(results)


if __name__ == '__main__':
    main()