import argparse
import torch.nn
from sympy import discriminant
from torch.utils.data import DataLoader
from torch.optim import AdamW
from utils import *
from dataset.mde_datasets import MDE_dataset
from dataset.mde_datasets_semi import MDE_dataset_semi
from torchmetrics.image import StructuralSimilarityIndexMeasure
import albumentations as A
from albumentations.core.composition import Compose
import segmentation_models_pytorch as smp
from depth_anything_v2.dpt import DepthAnythingV2
import torch.nn.functional as F
from validation import *
from discriminator import MLPDiscriminator
import torch.optim as optim

parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')

parser.add_argument('--json_path', default='/home/hao/hao/2025_june/MDE/src/create_dataset_spie/spie_mde_json_0_100.json', type=str)
parser.add_argument('--name', dest='name', type=str, default='spie_518_0_100_semi_dann_with_cosine')
parser.add_argument('--max_depth', default=100, type=float)
parser.add_argument('--device', default=1, type=int)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--no_zero_grad', action='store_true', help='Whether to zero optimizer gradients before backward pass.')

parser.add_argument('--encoder', default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--min_depth', default=1, type=float)
parser.add_argument('--bs', default=16, type=int)
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
            A.Resize(args.height, args.width, mask_interpolation=cv2.INTER_AREA),
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
            A.ToTensorV2(),
        ], seed=42)

    train_transform_w = Compose([
        A.Resize(args.height, args.width, mask_interpolation=cv2.INTER_AREA),
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ], seed=42)

    train_transform_s = Compose([
        A.AutoContrast(p=0.2),
        A.GaussianBlur(p=0.2),
        A.MedianBlur(blur_limit=15, p=0.2),
        A.RandomGamma(p=0.2),
        A.Defocus(p=0.2),
        A.RandomFog(alpha_coef=0.1, p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2(),
    ], seed=42)

    train_transform_norm = Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2(),
    ], seed=42)

    seed_worker, g = get_dataloader_seed_utils(seed=42)
    trainset = MDE_dataset_semi(args.json_path, mode='train',
                                transform=train_transform, transform_s=train_transform_s, transform_w=train_transform_w, transform_norm=train_transform_norm)
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4,
                             drop_last=False, shuffle=True, worker_init_fn=seed_worker,generator=g)

    valset_render = MDE_dataset(args.json_path, mode='val_render',
                                transform=Compose([
                                    A.Resize(args.height, args.width, mask_interpolation=cv2.INTER_AREA),
                                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    A.ToTensorV2()], seed=42))
    valloader_render = DataLoader(valset_render, batch_size=args.bs, pin_memory=True,
                                  num_workers=4, drop_last=True, worker_init_fn=seed_worker,generator=g)

    valset_real = MDE_dataset(args.json_path, mode='val_real',
                              transform=Compose([
                                  A.Resize(args.height, args.width),
                                  A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                  A.ToTensorV2()], seed=42, additional_targets={'valid_mask': 'mask'}))
    valloader_real = DataLoader(valset_real, batch_size=args.bs, pin_memory=True,
                                num_workers=4, drop_last=True, worker_init_fn=seed_worker,generator=g)


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
    # criterion_l1 = masked_l1_loss()

    optimizer = AdamW(
        [{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
         {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name],
          'lr': args.lr * 10.0}],
        lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)


    discriminator = MLPDiscriminator(input_dim=model_configs[args.encoder]['out_channels'][-1], hidden_dims=[256, 128, 64]).cuda(args.device)
    adversarial_loss = nn.BCEWithLogitsLoss()
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)

    total_iters = args.epochs * len(trainloader)

    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100,
                     'log10': 100, 'silog': 100}

    best_rmse = float('inf')  # initialize best RMSE
    for epoch in range(args.epochs):
        start_logging_epoch(args, epoch, previous_best)
        model.train()
        total_loss = 0

        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            if not args.no_zero_grad:
                optimizer_D.zero_grad()

            img, depth, valid_mask = sample['image'].cuda(args.device), sample['depth'].cuda(args.device), sample['valid_mask'].cuda(args.device)
            real_w = sample['real_w'].cuda(args.device)
            real_s = sample['real_s'].cuda(args.device)
            real_w_mask = sample['real_w_mask'].cuda(args.device)
            real_w_mask = (real_w_mask == 255).float()

            pred, features_pred = model(img, return_features=True)
            pred_w, features_pred_w = model(real_w, return_features=True)
            with torch.no_grad():
                pred_s, features_pred_s = model(real_s, return_features=True)


            # ---- Discriminator update ----
            discriminator.requires_grad_(True)
            pred_real = discriminator(features_pred.detach())  # real = 1
            pred_fake = discriminator(features_pred_w.detach())  # fake = 0

            loss_real = adversarial_loss(pred_real, torch.ones_like(pred_real))
            loss_fake = adversarial_loss(pred_fake, torch.zeros_like(pred_fake))
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()

            # ----- Generator adversarial loss -----
            discriminator.requires_grad_(False)
            pred_fake_g = discriminator(features_pred_w)  # no detach
            loss_adv = adversarial_loss(pred_fake_g, torch.ones_like(pred_fake_g))


            # ----- Other losses -----
            feat_pred = F.normalize(features_pred, dim=1)  # [B, 768]
            feat_pred_w = F.normalize(features_pred_w, dim=1)  # [B, 768]
            cos_sim = F.cosine_similarity(feat_pred, feat_pred_w, dim=1)
            loss_feat = 1 - cos_sim.mean()
            loss_sup_si = criterion(pred, depth, (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth))
            loss_sup = loss_sup_si + loss_feat


            mask = (real_w_mask == 1) & (pred_w >= args.min_depth) & (pred_w <= args.max_depth)

            #loss_hist = cdf_wasserstein_loss(pred_w.detach(), pred, mask_real=mask, mask_render=valid_mask)


            print("mask valid count:", mask.sum().item(), "mask shape:", mask.shape)
            print("pred_w min/max:", pred_w.min().item(), pred_w.max().item())
            print("pred_s min/max:", pred_s.min().item(), pred_s.max().item())


            loss_con_l1 = masked_l1_loss(pred_s, pred_w.detach(), mask.float())
            #loss_con_si = criterion(pred_s, pred_w.detach(), (real_w_mask == 1) & (pred_w >= args.min_depth) & (pred_w <= args.max_depth))
            #loss_con = loss_con_l1

            loss = loss_sup + loss_adv
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters = epoch * len(trainloader) + i

            lr = args.lr * (1 - iters / total_iters) ** 0.9

            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0

            logging.info(
                'Iter: {}/{}, LR: {:.7f}, Loss: {:.3f} (sup: {:.3f}, feat: {:.3f}, con_l1: {:.3f}, adv: {:.3f}, D: {:.3f})'.format(
                    i, len(trainloader),
                    optimizer.param_groups[0]['lr'],
                    loss.item(),
                    loss_sup_si.item(),
                    loss_feat.item(),
                    loss_con_l1.item(),
                    loss_adv.item(),
                    loss_D.item(),
                )
            )
        previous_best, best_rmse = validation_process(args, epoch, model, optimizer, valloader_render, valloader_real, previous_best, best_rmse)



if __name__ == '__main__':
    init_seeds(42)
    main()
