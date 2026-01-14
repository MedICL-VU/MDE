import os, torch, logging
from torch.autograd import Function
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

def save_checkpoint(net, save_dir, epoch, net_dict=None, best=False):
    os.makedirs(save_dir, exist_ok=True)
    save_checkpoint_dir = os.path.join(save_dir, 'cp')
    os.makedirs(save_checkpoint_dir, exist_ok=True)

    if best:
        torch.save(net.state_dict(), os.path.join(save_checkpoint_dir, 'best_net.pth'))
        logging.info(f'best Checkpoint net {epoch + 1} saved !')
    else:
        torch.save(net_dict, os.path.join(save_checkpoint_dir, 'last_net.pth'))
        logging.info(f'last Checkpoint net {epoch + 1} saved !')
        
        
def dice_coefficient_multiclass_batch(preds, targets, num_classes, epsilon=1e-6):

    preds = preds.squeeze(1)
    targets = targets.squeeze(1)
    preds_one_hot = torch.nn.functional.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2)
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2)

    preds_flat = preds_one_hot.view(preds_one_hot.shape[0], num_classes, -1)
    targets_flat = targets_one_hot.view(targets_one_hot.shape[0], num_classes, -1)

    intersection = torch.sum(preds_flat * targets_flat, dim=2)
    union = torch.sum(preds_flat, dim=2) + torch.sum(targets_flat, dim=2)

    dice = (2. * intersection + epsilon) / (union + epsilon)

    return dice


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.00001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda(input.get_device()).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


