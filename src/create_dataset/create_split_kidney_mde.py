import os
import json
import imageio.v3 as iio
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import random
import natsort
import glob
from pathlib import Path
import re, shutil

split = {'train': [], 'val': []}


data_dir = '/home/hao/hao/2025_june/data/mdephantom2'
video_lists = ['15L', '31R', '32R', '34L']

fakeB_dir = '/home/hao/hao/2025_june/data/mdephantom2/fake_endoscopy'
max_list = []
for video in video_lists:
    depth_dir = os.path.join(data_dir, video, 'raw_np')
    image_dir = os.path.join(fakeB_dir, video)

    depth_lists = sorted(os.listdir(depth_dir))
    image_lists = sorted(os.listdir(image_dir))

    assert len(depth_lists) == len(image_lists)
    for i in range(0, len(image_lists)):
        fakeB_path = os.path.join(fakeB_dir, video, image_lists[i])
        depth_path = os.path.join(depth_dir, depth_lists[i])

        data_pair = {'image': fakeB_path, 'depth': depth_path}

        if video == '15L':
            split['val'].append(data_pair)
        else:
            split['train'].append(data_pair)
        depth = np.load(depth_path)
        max_list.append(np.max(depth))
        # depth = depth * 100
        depth[depth>30] = 0
        np.save(depth_path, depth)
        print(np.max(depth), np.min(depth))

print(np.max(max_list), np.min(max_list))
    # fakeB_path = os.path.join(fakeB_dir, video, image_lists[178])
    # depth_path = os.path.join(depth_dir, depth_lists[178])
    #
    # fakeB = iio.imread(fakeB_path)
    #
    #
    # depth = np.load(depth_path)
    # depth_resized = resize(depth, fakeB.shape[:2], preserve_range=True, anti_aliasing=False)
    #
    # # Plot side by side
    # fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    #
    # axes[0].imshow(fakeB)
    # axes[0].set_title("fakeB Image")
    # axes[0].axis('off')
    #
    # im = axes[1].imshow(depth_resized, cmap='inferno')
    # axes[1].set_title("Resized Depth Map")
    # axes[1].axis('off')
    #
    # plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    # plt.tight_layout()
    # plt.show()
    # print(1)


output_path = 'kidney_mde_david.json'
with open(output_path, 'w') as f:
    json.dump(split, f, indent=2)
