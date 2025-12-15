import argparse
import torch.nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from utils import *
from dataset.mde_datasets import MDE_dataset
import albumentations as A
from albumentations.core.composition import Compose
import segmentation_models_pytorch as smp
from depth_anything_v2.dpt import DepthAnythingV2
from validation import *


parser = argparse.ArgumentParser(description='Depth Anything V2 for Relative Depth Estimation')
parser.add_argument('--json_path', default='/home/hao/hao/2025_june/MDE/src/create_dataset/kidney_mde_david_ft.json', type=str)
parser.add_argument('--name', dest='name', type=str, default='spie_518_0_100_vitl_plain_david_ft')
parser.add_argument('--device', default=1, type=int)
parser.add_argument('--epochs', default=5, type=int)
# parser.add_argument('--min_depth', default=0.01, type=float)
# parser.add_argument('--max_depth', default=100, type=float)
parser.add_argument('--min_depth', default=0.01, type=float)
parser.add_argument('--max_depth', default=30, type=float)
parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--bs', default=8, type=int)
parser.add_argument('--lr', default=0.000005, type=float)
parser.add_argument('--base_dir', type=str, default='checkpoints', help='Base dir to save checkpoints and images')
parser.add_argument('--height', type=int, default=518)
parser.add_argument('--width', type=int, default=518)



def main():
    args = parser.parse_args()
    args.save_path = os.path.join(args.base_dir, args.name, 'cp')
    os.makedirs(args.save_path, exist_ok=True)
    setup_logging(args, mode='train')
    logging.info(os.path.abspath(__file__))
    logging.info(args)


    train_transform = Compose([
            # A.Resize(args.height, args.width, mask_interpolation=cv2.INTER_AREA),
            ResizeWithSeparateMaskModes(args.height, args.width),
            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.Affine((0.7, 1.3), {'x': (-0.3, 0.3), 'y': (-0.3, 0.3)}, rotate=(-360, 360), p=0.25),
            A.GaussianBlur(p=0.1),
            A.AutoContrast(p=0.1),
            A.MedianBlur(blur_limit=15, p=0.1),
            A.RandomGamma(p=0.1),
            A.Defocus(p=0.1),
            A.RandomFog(alpha_coef=0.1, p=0.1),
            A.RandomBrightnessContrast(p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.ToTensorV2()
        ], seed=42)
    seed_worker, g = get_dataloader_seed_utils(seed=42)

    trainset = MDE_dataset(args.json_path, mode='train', transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=False, shuffle=True, generator=g)

    valset_render = MDE_dataset(args.json_path, mode='val',
                                transform=Compose([
                                    #A.Resize(args.height, args.width, mask_interpolation=cv2.INTER_AREA),
                                    ResizeWithSeparateMaskModes(args.height, args.width),
                                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    A.ToTensorV2()], seed=42))
    valloader_render = DataLoader(valset_render, batch_size=args.bs, pin_memory=True,
                                  num_workers=4, drop_last=True, worker_init_fn=seed_worker, generator=g)




    model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
    model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth}).cuda(args.device)


    if 'endoomni' in args.name:
        logging.info('endoomni')
        if 'vitl' in args.name:
            pretrained_path = '/home/hao/hao/2025_june/MDE/src/depth_anything_v2/EndoOmni_l.pt'
        elif 'vitb' in args.name:
            pretrained_path = '/home/hao/hao/2025_june/MDE/src/depth_anything_v2/EndoOmni_b.pt'
        elif 'vits' in args.name:
            pretrained_path = '/home/hao/hao/2025_june/MDE/src/depth_anything_v2/EndoOmni_s.pt'
        full_state_dict = torch.load(pretrained_path, map_location='cpu')['model']
    else:
        pretrained_path = f'/home/hao/hao/2025_june/MDE/src/depth_anything_v2/depth_anything_v2_metric_hypersim_{args.encoder}.pth'
        full_state_dict = torch.load(pretrained_path, map_location='cpu')
    # a = torch.load(pretrained_path, map_location='cpu')['model']
    # model.load_state_dict(
    #     {k: v for k, v in torch.load(pretrained_path, map_location='cpu').items() if 'pretrained' in k}, strict=False)

    filtered_state_dict = {k: v for k, v in full_state_dict.items() if 'pretrained' in k}
    logging.info("Loaded keys:")
    for k in filtered_state_dict.keys():
        logging.info(k)

    # Load into model
    model.load_state_dict(filtered_state_dict, strict=False)
    result = model.load_state_dict(filtered_state_dict, strict=False)

    logging.info("Missing keys:")
    for k in result.missing_keys:
        logging.info(f"  {k}")

    logging.info("Unexpected keys:")
    for k in result.unexpected_keys:
        logging.info(f"  {k}")

    criterion = SiLogLoss().cuda(args.device)
    criterion_l1 = torch.nn.L1Loss().to(args.device)

    optimizer = AdamW(
        [{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
         {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name],
          'lr': args.lr * 10.0}],
        lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    total_iters = args.epochs * len(trainloader)

    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100,
                     'log10': 100, 'silog': 100}
    best_abs_rel = float('inf')  # initialize best abs_rel

    for epoch in range(args.epochs):
        start_logging_epoch(args, epoch, previous_best)

        model.train()
        total_loss = 0

        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            img, depth, valid_mask = sample['image'].cuda(args.device), sample['depth'].cuda(args.device), sample['valid_mask'].cuda(args.device)

            pred = model(img).squeeze(1)

            loss = criterion(pred, depth, (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters = epoch * len(trainloader) + i

            lr = args.lr * (1 - iters / total_iters) ** 0.9

            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0

            if i % 1 == 0:
                logging.info(
                    'Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'],
                                                                   loss.item()))

        previous_best, best_abs_rel = validation_process_no_real(args, epoch, model, optimizer, valloader_render,
                                                      previous_best, best_abs_rel)


if __name__ == '__main__':
    init_seeds(42)
    main()
