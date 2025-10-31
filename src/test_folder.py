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
import tifffile



if __name__ == '__main__':
    a = time.time()
    init_seeds(42)
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')

    #pretrained_path = '/home/hao/hao/2025_june/MDE/src/checkpoints/spie_518_0_100_vits_plain_cv3d_relative/cp/latest.pth'
    #pretrained_path = '/home/hao/hao/2025_june/MDE_registration/src/checkpoints_hackathon/mde_518_bph/cp/mde_bph_518_100.pth'
    pretrained_path = '/home/hao/hao/2025_june/MDE/src/checkpoints/spie_518_0_100_semi_dann_with_cosine_endoomni_vitl/cp/latest_endoL.pth'
    max_depth = 100
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    net = DepthAnythingV2(**{**model_configs['vitl'], 'max_depth': max_depth}).to(device)

    net.load_state_dict(torch.load(pretrained_path, map_location=device)['model'])

    save_dir = pretrained_path.replace('.pth', '_results')
    json_file = '/home/hao/hao/2025_june/MDE/src/create_dataset/split_cv3d_with_test.json'
    import json
    with open(json_file, 'r') as f:
        image_list = json.load(f)['test']

    image_dir = '/home/hao/hao/mde_ct_camera/mcap_to_vision/images_cao'
    image_list = sorted(os.listdir(image_dir))
    save_dir = '/home/hao/hao/mde_ct_camera/mcap_to_vision/depths_cao'
    image_size = 518
    os.makedirs(save_dir, exist_ok=True)


    transform = Compose([Resize(image_size, image_size),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                         ])

    transform_resize = Compose([Resize(image_size, image_size),
                         ], additional_targets={'depth': 'mask'})

    mae_list = []
    for i in range(0, len(image_list)):
        image_path = os.path.join(image_dir, image_list[i])

        try:
            img = Image.open(image_path)
        except:
            print(image_path)
            continue
        original_image = np.array(img)  # H, W = image.shape[0], image.shape[1]

        image_transformed = transform(image=original_image)['image']

        resized = transform_resize(image=original_image)

        image_resize = resized['image']

        from rembg import remove
        if not os.path.exists('./masj.npy'):
            circle_mask = remove(image_resize, only_mask=True)  # uint8 [0,255], shape HxW
            if circle_mask.ndim == 3:
                circle_mask = cv2.cvtColor(circle_mask, cv2.COLOR_BGR2GRAY)

            _, binary_mask = cv2.threshold(circle_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = (binary_mask > 0).astype(np.float32)  # HxW float in {0,1}
            np.save('./masj.npy', mask)
        else:
            mask = np.load('./masj.npy')

        image_transformed = np.moveaxis(image_transformed, -1, 0)
        input_image = torch.from_numpy(image_transformed).to(device=device).float().unsqueeze(0)
        with torch.cuda.amp.autocast():
            pred = net(input_image)
        predict_mask_numpy = pred[0, :].detach().cpu().numpy()
        predict_mask_numpy = predict_mask_numpy * mask
        print(np.unique(predict_mask_numpy))


        save_name = image_path.split('/')[-1]
        # save_path = os.path.join(save_dir, video_name, save_name.replace('.png', '_pred'))
        save_path = os.path.join(save_dir, save_name.replace('.png', '_pred'))
        np.save(save_path, predict_mask_numpy)


    print(f'mean MAE: {np.mean(mae_list)}')
    print(len(mae_list))
    print(time.time() - a)

