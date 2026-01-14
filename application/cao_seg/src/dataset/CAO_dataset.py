import os
import torch
import numpy as np
from torch.utils.data import Dataset
import glob, json
import PIL.Image as Image
import random

filename_types = ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'PNG', 'JPG', 'JPEG', 'TIFF', 'TIFF', 'npz', 'npy']

class CAO_dataset(Dataset):
    def __init__(self, args, mode='train', transform=None):
        self.args = args
        self.mode = mode
        if mode != 'test':
            json_file_path = args.json_path
            with open(json_file_path, 'r') as file:
                split = json.load(file)
            self.data = split[mode]
        else:
            data = sorted(os.listdir(args.test_data_dir))
            self.data = [f for f in data if f != ".DS_Store"]
        self.transform = transform


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        if self.mode != 'test':
            image_path = self.data[idx]['image']
            label_path = self.data[idx]['label']
            img = Image.open(image_path)        # [3, Height, Width]
            if img.mode == "RGBA":
                image = np.array(img.convert("RGB"))
            mask = np.array(Image.open(label_path).convert('L'))
            mask[mask > 0] = 1
            assert len(np.unique(mask)) == 2, f"Mask unique values: {np.unique(mask)}"
            h, w = image.shape[:2]
        else:
            image_path = os.path.join(self.args.test_data_dir, self.data[idx])
            if '.npy' in image_path:
                image = np.load(image_path)
            else:
                img = Image.open(image_path)        # [3, Height, Width]
                if img.mode == "RGBA":
                    image = np.array(img.convert("RGB"))
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

            image = image[:, 320:1660, :]
            mask = mask[:, 320:1660]

        mask = np.expand_dims(mask, -1)

        augmented = self.transform(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']

        image = np.moveaxis(image, -1, 0)
        mask = np.moveaxis(mask, -1, 0)

        data = {
                  'image': torch.from_numpy(image).type(torch.FloatTensor),
                  'label': torch.from_numpy(mask).type(torch.FloatTensor),
                  'name': image_path,
                  'original_shape': (h, w)
                }

        return data



