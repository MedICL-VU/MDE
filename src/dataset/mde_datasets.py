import os
from pickletools import uint8

import torch
import numpy as np
from torch.utils.data import Dataset
import glob
import PIL.Image as Image
import json
import tifffile
filename_types = ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'PNG', 'JPG', 'JPEG', 'TIFF', 'TIFF', 'npz', 'npy']
class MDE_dataset(Dataset):
    def __init__(self, json_path, mode='train', transform=None, geometry_transform=None, norm_transform=None):
        self.transform = transform
        if mode != 'test':
            json_file_path = json_path
            with open(json_file_path, 'r') as file:
                split = json.load(file)

            if mode == 'train' and 'render' in split.get('train', {}):
                self.data = split['train']['render']
            else:
                self.data = split[mode]
        #self.data = self.data[0:10]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]['image']
        label_path = self.data[idx]['depth']

        # === Load image ===
        image = Image.open(image_path)
        if image.mode == "RGBA":
            image = np.array(image.convert("RGB"))
        else:
            image = np.array(image)

        height, width = image.shape[:2]

        # === Load depth ===
        if label_path.endswith('.npz'):
            depth = np.load(label_path)['depth'].astype(np.float32)
        elif label_path.endswith('.npy'):
            depth = np.load(label_path)
        elif label_path.endswith('.tiff'):
            depth = tifffile.imread(label_path).astype(float)
            depth = depth / 65535 * 100
            valid_mask = depth > 0
            # depth[valid_mask] = 1000.0 / depth[valid_mask]
        else:
            depth = Image.open(label_path).convert('L')

        if image.shape[:2] != depth.shape[:2]:
            from skimage.transform import resize
            image = resize(image, depth.shape[:2], preserve_range=True, order=1).astype(np.uint8)

        image = image[1:-1, 1:-1]
        depth = depth[1:-1, 1:-1]



        # === Apply transforms ===
        if self.transform:
            augmented = self.transform(image=image, mask=depth)
            image = augmented['image']
            depth = augmented['mask']

        # === Valid mask: depth > 0 ===
        # valid_mask = torch.zeros_like(depth, dtype=torch.uint8)
        # valid_mask[depth > 0] = 1
        valid_mask = cv2.resize(valid_mask, (image.shape[2], image.shape[1]), interpolation=cv2.INTER_NEAREST)



        # === Convert image to CHW ===
        image = image * valid_mask[None, :, :]

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(image.detach().cpu().permute(1, 2, 0).numpy())
        # plt.title(image_path)
        # plt.figure()
        # plt.imshow(depth.detach().cpu().numpy())
        # plt.title(label_path)
        # plt.show()
        data = {
            'image': image,
            'depth': depth,
            'valid_mask': valid_mask,
            'name': image_path
        }
        return data






if __name__ == '__main__':
    data_folder = '/media/hao/mydrive1/MDE/data_training'
    dataset = MDE_dataset(data_folder, mode='train', transform=None)
    image, depth = dataset.__getitem__(0)['image'], dataset.__getitem__(0)['depth']
