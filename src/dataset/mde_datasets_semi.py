import os
from pickletools import uint8
import math
import torch
import numpy as np
from torch.utils.data import Dataset
import glob
import PIL.Image as Image
import json

filename_types = ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'PNG', 'JPG', 'JPEG', 'TIFF', 'TIFF', 'npz', 'npy']
class MDE_dataset_semi(Dataset):
    def __init__(self, json_path, mode='train', transform=None, transform_s=None, transform_w=None, transform_norm=None):
        self.transform = transform
        self.transform_s = transform_s
        self.transform_w = transform_w
        self.transform_norm = transform_norm

        if mode != 'test':
            json_file_path = json_path
            with open(json_file_path, 'r') as file:
                split = json.load(file)

            if mode == 'train':
                self.render_list, self.real_list = self.get_image_path_list(split)
                #self.render_list, self.real_list = self.render_list[:100], self.real_list[:100]
            # if mode == 'train':
            #     self.data = split[mode][0:100]
            # else:
            #     self.data = split[mode]

    def __len__(self):
        return len(self.render_list)

    def __getitem__(self, idx):
        image_path = self.render_list[idx]['image']
        label_path = self.render_list[idx]['depth']

        real_image_path = self.real_list[idx]
        # === Load image ===
        image = Image.open(image_path)
        real = Image.open(real_image_path)

        mask_path = os.path.join('/home/hao/hao/2025_june/MDE/src/create_dataset_spie', real_image_path.split('/')[-2]+'_mask.png')
        real_mask = Image.open(mask_path).convert('L')


        image = np.array(image)
        real = np.array(real)
        real_mask = np.array(real_mask)

        height, width = image.shape[:2]

        # === Load depth ===
        if label_path.endswith('.npz'):
            depth = np.load(label_path)['depth'].astype(np.float32)
        elif label_path.endswith('.npy'):
            depth = np.load(label_path)
        else:
            depth = Image.open(label_path).convert('L')



        if image.shape[:2] != depth.shape[:2]:
            from skimage.transform import resize
            image = resize(image, depth.shape[:2], preserve_range=True, order=1).astype(np.uint8)
            mask = depth > 0
            if image.ndim == 3:
                image = image * mask[..., None]  # for RGB
            else:
                image = image * mask  # for grayscale




        # === Apply transforms ===
        augmented = self.transform(image=image, mask=depth)
        image = augmented['image']
        depth = augmented['mask']

        augmented_w = self.transform_w(image=real, mask=real_mask)
        real_w, real_w_mask = augmented_w['image'], augmented_w['mask']
        real_s = self.transform_s(image=real_w)['image']

        augmented_w_norm = self.transform_norm(image=real_w, mask=real_w_mask)
        real_w, real_w_mask = augmented_w_norm['image'], augmented_w_norm['mask']
        # === Valid mask: depth > 0 ===
        valid_mask = np.zeros_like(depth, dtype=np.uint8)
        valid_mask[depth > 0] = 1

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(real_w.detach().cpu().numpy().transpose(1, 2, 0))
        # plt.title(real_image_path)
        # plt.show()
        # # plt.imshow(valid_mask)
        # plt.figure()
        # plt.imshow(real_s.detach().cpu().numpy().transpose(1, 2, 0))
        # plt.title(real_image_path)
        # plt.figure()
        # plt.imshow(real_w_mask.detach().cpu().numpy())
        # plt.title(mask_path)
        # plt.show()

        data = {
            'image': image,
            'depth': depth,
            'valid_mask': valid_mask,
            'name': image_path,
            'real_w': real_w,
            'real_s':real_s,
            'real_w_mask': real_w_mask,
            'name_real': real_image_path,

        }
        return data


    def get_image_path_list(self, split):
        filenames_render = split['train']['render']
        filenames_real = split['train']['real']

        repetition_factor = math.ceil(len(filenames_render) / len(filenames_real))
        filenames_real *= repetition_factor
        filenames_real = filenames_real[:len(filenames_render)]

        return filenames_render, filenames_real






if __name__ == '__main__':
    data_folder = '/media/hao/mydrive1/MDE/data_training'
    dataset = MDE_dataset_semi(data_folder, mode='train', transform=None)

    image, depth = dataset.__getitem__(0)['image'], dataset.__getitem__(0)['depth']
