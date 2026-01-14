import argparse
import os
import warnings
import logging
import datetime

parser = argparse.ArgumentParser()


# general setting
default_json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/get_split/split.json"))
parser.add_argument('--json_path', type=str, default=None)
parser.add_argument('--test_data_dir', type=str, default=None)

parser.add_argument('--mode', type=str, default='test')

parser.add_argument('--total_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)

parser.add_argument('--num_workers', type=int, default=2)

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--min_lr', type=float, default=1e-5)

parser.add_argument('--height', type=int, default=224)
parser.add_argument('--width', type=int, default=224)

default_sam2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../sam2_checkpoints/sam2_hiera_large.pt"))
parser.add_argument('--sam2_path', type=str, default=default_sam2_path)



parser.add_argument('--name', dest='name', type=str, default='test123')
default_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../checkpoints"))
parser.add_argument('--base_dir', type=str, default=default_base_dir, help='Base dir to save checkpoints and images')


# network setting
parser.add_argument('--net', type=str, default='unet',
                    help='Specify model type [unet | unet++ | densenet | unet3d | swin_unet]')
parser.add_argument('--attn', action='store_true', help='add attention modules')
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--save_results', action='store_true')
parser.add_argument('--save_results_dir', type=str, default='save_results')




parser.add_argument('--device', type=int, default=1)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--out_channels', type=int, default=1)








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



