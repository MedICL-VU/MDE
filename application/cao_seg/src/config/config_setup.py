import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.core.composition import Compose
from dataset.CAO_dataset import CAO_dataset
import random, os
import numpy as np
import logging
from torch import optim
from models.CATS2d import CATS2d

def get_net(args, pretrain=False, model=None, net=None):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple MPS (Mac M1/M2)
    else:
        device = torch.device("cpu")

    net = args.net
    logging.info(f'Using device {device}')
    logging.info(f'Building:  {net}')

    if net == 'unet':
        inference_mode = True if args.mode == 'test' else False
        net = smp.Unet(encoder_name='mobileone_s1', encoder_weights='imagenet',
                       in_channels=args.in_channels, classes=args.out_channels)
        net.encoder.inference_mode=inference_mode

    elif net == 'cats2d':
        net = CATS2d(checkpoint_path=args.sam2_path, device=device)

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logging.info(f"Total trainable parameters: {total_params}")

    if pretrain:
        pretrain_path = os.path.join(args.save_dir, 'cp', 'best_net.pth')
        net.load_state_dict(torch.load(pretrain_path, map_location=device))
        logging.info(f'Model{model}  loaded from {pretrain_path}')

    net.to(device=device)
    return net


def get_optimizer_and_scheduler(args, net):
    params = filter(lambda p: p.requires_grad, net.parameters())  # added from
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epoch, eta_min=args.min_lr)

    return optimizer, scheduler



def get_dataset(args, mode=None, json=False):
    if mode is None:
        raise ValueError('mode must be specified')

    if mode == 'train':
        transform = Compose([
            A.Resize(args.height, args.width),
            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine((0.7, 1.3), {'x': (-0.3, 0.3), 'y': (-0.3, 0.3)}, rotate=(-360, 360), p=0.25),
            A.GaussianBlur(p=0.1),
            A.AutoContrast(p=0.1),
            A.MedianBlur(blur_limit=15, p=0.1),
            A.RandomGamma(p=0.1),
            A.Defocus(p=0.1),
            A.RandomFog(alpha_coef=0.1, p=0.1),
            A.RandomBrightnessContrast(p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ], seed=42)
    else:
        transform = Compose([
            A.Resize(args.height, args.width),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ], seed=42)


    dataset_cao = CAO_dataset(args, mode=mode, transform=transform)

    return dataset_cao


def init_seeds(seed=42, cuda_deterministic=True):
    from torch.backends import cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True
