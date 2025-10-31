import random
import numpy as np
import torch
import os
import logging
import datetime
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim


def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(),
            'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item()}


def normalized_cross_correlation_1d(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    return numerator / (denominator + 1e-8)


def compute_masked_ssim(pred, target, mask):
    y, x = np.where(mask > 0)
    if len(y) == 0 or len(x) == 0:
        return float('nan')  # or 0.0 if preferred

    y_min, y_max = y.min(), y.max()
    x_min, x_max = x.min(), x.max()

    pred_crop = pred[y_min:y_max+1, x_min:x_max+1]
    target_crop = target[y_min:y_max+1, x_min:x_max+1]
    mask_crop = mask[y_min:y_max+1, x_min:x_max+1]

    pred_crop = pred_crop * mask_crop
    target_crop = target_crop * mask_crop
    return ssim(pred_crop, target_crop, data_range=target_crop.max() - target_crop.min())


def eval_depth_with_mae(pred, target):
    pred = pred.astype(np.float32)

    pred = np.clip(pred, 1e-6, None)
    target = np.clip(target, 1e-6, None)

    thresh = np.maximum(target / pred, pred / target)

    d1 = np.mean(thresh < 1.25)
    d2 = np.mean(thresh < 1.25 ** 2)
    d3 = np.mean(thresh < 1.25 ** 3)

    diff = pred - target
    diff_log = np.log(pred) - np.log(target)

    abs_rel = np.mean(np.abs(diff) / target)
    sq_rel = np.mean((diff ** 2) / target)
    rmse = np.sqrt(np.mean(diff ** 2))
    rmse_log = np.sqrt(np.mean(diff_log ** 2))
    log10 = np.mean(np.abs(np.log10(pred) - np.log10(target)))
    silog = np.sqrt(np.mean(diff_log ** 2) - 0.5 * (np.mean(diff_log) ** 2))
    mae = np.mean(np.abs(diff))

    ncc = normalized_cross_correlation_1d(pred, target)
    return {
        'd1': d1, 'd2': d2, 'd3': d3,
        'abs_rel': abs_rel, 'sq_rel': sq_rel,
        'rmse': rmse, 'rmse_log': rmse_log,
        'log10': log10, 'silog': silog,
        'mae': mae, 'ssim': 0, 'ncc': ncc
    }


# class SiLogLoss(nn.Module):
#     def __init__(self, lambd=0.5):
#         super().__init__()
#         self.lambd = lambd
#
#     def forward(self, pred, target, valid_mask):
#         valid_mask = valid_mask.detach()
#         diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
#         loss = torch.sqrt(torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2))
#
#         return loss


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5, epsilon=1e-6):
        super().__init__()
        self.lambd = lambd
        self.epsilon = epsilon

    def forward(self, pred, target, valid_mask):
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        pred = pred.clamp(min=self.epsilon)
        target = target.clamp(min=self.epsilon)

        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])

        mean_squared = diff_log.pow(2).mean()
        mean_mean = diff_log.mean().pow(2)

        val = mean_squared - self.lambd * mean_mean
        val = torch.clamp(val, min=0.0)  # ✅ prevent sqrt of negative

        return torch.sqrt(val)



def compute_normalized_hist(depth, mask, bins=None, range=None):
    depth = torch.clamp(depth, min=1e-3)
    if mask is not None:
        depth = depth[mask > 0]
    else:
        depth = depth.view(-1)

    log_depth = torch.log(depth)
    mean = log_depth.mean()
    std = log_depth.std() + 1e-6
    log_norm = (log_depth - mean) / std  # z-score

    hist = torch.histc(log_norm, bins=bins, min=range[0], max=range[1])
    hist = hist / (hist.sum() + 1e-6)  # normalize to prob dist
    return hist






def cdf_wasserstein_loss(pred_real, pred_render, mask_real=None, mask_render=None, bins=64, range=(-3, 3), sigma=0.1):
    """
    soft version of cdf-wasserstein loss
    """
    def get_soft_cdf(depth, mask):
        depth = torch.clamp(depth, min=1e-3)
        if mask is not None:
            depth = depth[mask > 0]
        else:
            depth = depth.view(-1)

        log_d = torch.log(depth)
        log_d = (log_d - log_d.mean()) / (log_d.std() + 1e-6)

        bin_centers = torch.linspace(range[0], range[1], bins, device=log_d.device)
        bin_centers = bin_centers.view(1, -1)  # [1, B]
        log_d = log_d.view(-1, 1)  # [N, 1]

        weights = torch.exp(-0.5 * ((log_d - bin_centers) / sigma) ** 2)
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-6)  # normalize per bin
        hist = weights.sum(dim=0)
        hist = hist / (hist.sum() + 1e-6)
        cdf = torch.cumsum(hist, dim=0)
        return cdf

    cdf_real = get_soft_cdf(pred_real, mask_real)
    cdf_render = get_soft_cdf(pred_render, mask_render)

    return torch.mean(torch.abs(cdf_real - cdf_render))


def depth_moment_alignment(real_preds, render_preds, real_mask, render_mask):
    def get_log_moments(d, m):
        log_d = torch.log(torch.clamp(d[m], min=1e-3))
        return log_d.mean(), log_d.std()

    m1, s1 = get_log_moments(real_preds, real_mask)
    m2, s2 = get_log_moments(render_preds, render_mask)
    return torch.abs(m1 - m2) + torch.abs(s1 - s2)

















def setup_logging(args, mode='train'):
    log_dir = os.path.join(args.base_dir, args.name, 'logs', mode)
    os.makedirs(log_dir, exist_ok=True)

    # Get the current time and format it for the log file name
    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    mode_prefix = 'training' if mode == 'train' else 'test'
    log_file = os.path.join(log_dir, f'{mode_prefix}_{start_time}.log')

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )

    args.save_dir = os.path.join(args.base_dir, args.name)
    logging.info("Save directory is: {}".format(args.save_dir))
    logging.info(f"Logging setup complete. Logs will be saved in {log_file}")


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

def masked_l1_loss(pred, target, mask):
    """
    pred, target: [B, 1, H, W] or [B, H, W]
    mask: boolean or float tensor of same shape, where 1 indicates valid pixels
    """
    abs_diff = torch.abs(pred - target)
    masked_loss = abs_diff * mask
    return masked_loss.sum() / (mask.sum() + 1e-6)



def start_logging_epoch(args, epoch, previous_best):
    logging.info('===========> Epoch: {:}/{:}, d1: {:.3f}, d2: {:.3f}, d3: {:.3f}'.format(epoch, args.epochs,
                                                                                          previous_best['d1'],
                                                                                          previous_best['d2'],
                                                                                          previous_best['d3']))
    logging.info('===========> Epoch: {:}/{:}, abs_rel: {:.3f}, sq_rel: {:.3f}, rmse: {:.3f}, rmse_log: {:.3f}, '
                 'log10: {:.3f}, silog: {:.3f}'.format(
        epoch, args.epochs, previous_best['abs_rel'], previous_best['sq_rel'], previous_best['rmse'],
        previous_best['rmse_log'], previous_best['log10'], previous_best['silog']))


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


def get_dataloader_seed_utils(seed=42):
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return seed_worker, g



def compute_gradients(depth):
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]  # vertical gradient
    dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]  # horizontal gradient
    return dx, dy

def gradient_consistency_loss_masked(depth1, depth2, mask):
    """
    depth1, depth2: (B, 1, H, W)
    mask: (B, 1, H, W) - binary or float mask (1 = valid)
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    # Intersect mask for both inputs and adjust for gradient shape
    mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]  # (B, 1, H-1, W)
    mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]  # (B, 1, H, W-1)

    dx1, dy1 = compute_gradients(depth1)
    dx2, dy2 = compute_gradients(depth2)

    diff_dx = torch.abs(dx1 - dx2) * mask_x
    diff_dy = torch.abs(dy1 - dy2) * mask_y

    eps = 1e-6  # prevent division by zero
    loss_dx = diff_dx.sum() / (mask_x.sum() + eps)
    loss_dy = diff_dy.sum() / (mask_y.sum() + eps)

    return loss_dx + loss_dy


def image_gradient_alignment_loss(depth, image, mask):
    """
    Align depth gradients with image gradients.

    depth: (B, 1, H, W) - predicted depth
    image: (B, 3, H, W) - input RGB image
    mask: (B, 1, H, W) - binary mask of valid pixels
    """

    # Convert image to grayscale
    gray = 0.2989 * image[:, 0:1] + 0.5870 * image[:, 1:2] + 0.1140 * image[:, 2:3]

    # Compute gradients
    depth_dx, depth_dy = compute_gradients(depth)
    gray_dx, gray_dy = compute_gradients(gray)
    mask_dx, mask_dy = compute_gradients(mask)

    # Weight: high if image has strong gradient, low otherwise
    weight_x = torch.exp(-10 * torch.abs(gray_dx))
    weight_y = torch.exp(-10 * torch.abs(gray_dy))

    # Compute loss (masked)
    loss_x = weight_x * torch.abs(depth_dx)
    loss_y = weight_y * torch.abs(depth_dy)

    # Apply mask (shrinked for gradient)
    valid_mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    valid_mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    eps = 1e-6
    loss = (
            (loss_x * valid_mask_x).sum() / (valid_mask_x.sum() + eps) +
            (loss_y * valid_mask_y).sum() / (valid_mask_y.sum() + eps)
    )

    return loss
