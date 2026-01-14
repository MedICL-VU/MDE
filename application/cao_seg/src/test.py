import os.path

from dataset.CAO_dataset import CAO_dataset
from torch.utils.data import DataLoader
from albumentations.core.composition import Compose
from albumentations import Resize
import numpy as np
import torch
from tqdm import tqdm
from util.utils import dice_coefficient_multiclass_batch
from config.config_args import *
from config.config_setup import get_net, init_seeds
from PIL import Image
from config.config_setup import get_dataset
import cv2
import os
import sys
sys.path = list(dict.fromkeys(sys.path))
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
project_root = os.path.dirname(os.getcwd())
if project_root in sys.path:
    sys.path.remove(project_root)

def validate_baseline(args, net1, loader, save_results_dir=None):
    if args.save_results:
        if not os.path.exists(save_results_dir):
            os.makedirs(save_results_dir, exist_ok=True)
    net1.eval()
    dice_list1 = []

    with torch.no_grad():
        with tqdm(total=len(loader), desc='Validation round', unit='batch', leave=False) as pbar:
            for i, (batch) in enumerate(loader):
                input, target = batch['image'].to(device=args.device, dtype=torch.float32), batch['label'].to(device=args.device, dtype=torch.long)

                if args.net == 'cats2d':
                    output1, _, _ = net(input)
                else:
                    output1 = net(input)

                output1_prob = torch.sigmoid(output1)
                pred_labels = (output1_prob > 0.5).float()
                pred_labels = pred_labels.unsqueeze(1).detach().cpu()

                if args.save_results:
                    frame_name = batch['name'][0].split('/')[-1]
                    predict_mask_numpy = pred_labels.cpu().numpy().squeeze()
                    resized_array = cv2.resize(predict_mask_numpy, (1340, 1080), interpolation=cv2.INTER_NEAREST)
                    resized_array[resized_array == 1] = 255
                    blank_mask = np.zeros((1080, 1920))
                    blank_mask[:, 320:1660] = resized_array

                    # blank_mask_resized = cv2.resize(blank_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
                    # blank_mask_resized = blank_mask_resized.astype(np.uint8)
                    cv2.imwrite(os.path.join(save_results_dir, frame_name.replace('.npy', '.png')), blank_mask)
                    # np.save(os.path.join(save_results_dir, frame_name), blank_mask)


                pbar.update(1)
            pbar.close()
        return dice_list1


def test_net_baseline(args, net1, dataset, batch_size=1):
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    logging.info(f'''Starting testing:
            Num test:        {len(dataset)}
            Batch size:      {batch_size}
            Device:          {torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')}
        ''')

    net1.eval()

    dice_list1 = validate_baseline(args, net1, test_loader, save_results_dir=args.test_data_dir.replace('/image/', f'/seg_{args.name}/'))
    logging.info('Model, batch-wise validation Dice coeff: {}, std: {}'.format(np.mean(dice_list1), np.std(dice_list1)))




if __name__ == '__main__':
    init_seeds(42)
    args = parser.parse_args()
    setup_logging(args, mode='test')
    logging.info(os.path.abspath(__file__))
    logging.info(args)

    dataset = get_dataset(args, mode='test')
    net = get_net(args, pretrain=True)

    test_net_baseline(args,
             net1=net,
             dataset=dataset)

