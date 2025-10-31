from PIL.ImageOps import grayscale
from albumentations.core.composition import Compose
from albumentations import Resize, Normalize
import torch
from PIL import Image
import segmentation_models_pytorch as smp
import random
import numpy as np
import os, time, cv2
from depth_anything_v2.dpt import DepthAnythingV2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import *
from skimage.transform import resize

if __name__ == '__main__':
    a = time.time()
    init_seeds(42)
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')

    name = 'spie_518_0_100_semi_dann_with_cosine_endoomni_vitb'
    pretrained_path = f'/home/hao/hao/2025_june/MDE/checkpoints/{name}/cp/latest.pth'
    #pretrained_path = '/home/hao/hao/2025_june/MDE/src/checkpoints_hackathon/mde_518/cp/mde_518.pth'
    max_depth = 100
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    net = DepthAnythingV2(**{**model_configs['vitb'], 'max_depth': max_depth}).to(device)

    net.load_state_dict(torch.load(pretrained_path, map_location=device)['model'])

    save_dir = pretrained_path.replace('.pth', '_results')
    json_file = '/home/hao/hao/2025_june/MDE/src/create_dataset_spie/spie_mde_json_0_100.json'
    import json
    with open(json_file, 'r') as f:
        image_list = json.load(f)['val_real']


    image_size = 518
    os.makedirs(save_dir, exist_ok=True)


    transform = Compose([Resize(image_size, image_size),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                         ])

    transform_resize = Compose([Resize(image_size, image_size),
                         ], additional_targets={'depth': 'mask'})

    mae_list = []
    for i in range(0, len(image_list)):
        image_path = image_list[i]['image']
        depth_path = image_list[i]['depth']


        try:
            img = Image.open(image_path)

            if depth_path.endswith('.npz'):
                depth = np.load(depth_path)['depth'].astype(np.float32)
            if depth_path.endswith('.npy'):
                depth = np.load(depth_path)
            else:
                depth = Image.open(depth_path).convert('L')
        except:
            print(image_path)
            continue
        original_image = np.array(img)  # H, W = image.shape[0], image.shape[1]
        original_depth = np.array(depth)

        if original_image.shape[:2] != original_depth.shape[:2]:
            original_image = resize(original_image, original_depth.shape[:2], preserve_range=True, order=1).astype(np.uint8)

        image_transformed = transform(image=original_image)['image']

        resized = transform_resize(image=original_image, depth=original_depth)
        image_resize = resized['image']
        depth_transformed = resized['depth']
        mask = depth_transformed>0

        # import matplotlib.pyplot as plt
        # plt.imshow(original_depth)
        # plt.show()

        image_transformed = np.moveaxis(image_transformed, -1, 0)
        input_image = torch.from_numpy(image_transformed).to(device=device).float().unsqueeze(0)
        with torch.cuda.amp.autocast():
            pred = net(input_image)
        predict_mask_numpy = pred[0, :].detach().cpu().numpy()
        predict_mask_numpy = predict_mask_numpy * mask


        diff = abs(depth_transformed - predict_mask_numpy)
        predict_mask_numpy_mask = predict_mask_numpy
        masked_diff = diff * mask
        mae = masked_diff.sum() / mask.sum()
        print(mae)
        mae_list.append(mae)

        inverse_transform = Compose([Resize(image_size, image_size)])
        save_name = image_path.split('/')[-1]
        save_path = os.path.join(save_dir, save_name)


    print(f'mean MAE: {np.mean(mae_list)}')
    print(len(mae_list))
    print(time.time() - a)

