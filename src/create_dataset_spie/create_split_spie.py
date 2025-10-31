import os
import json
import imageio.v3 as iio
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import random


split = {'train': {'render': [], 'real': []}, 'val_render': [], 'val_real': []}


unity_data_dir = '/home/hao/hao/2025_june/unity_results_masked'
unity_data_dir1 = '/home/hao/hao/0_100_unity_data/cao'

cut_dirs = [
    '/home/hao/hao/CUT_results/cut_baseline_aug_512_BtoA_unity_0_100_cao/train_10/images/fake_B',
    '/home/hao/hao/CUT_results/cut_baseline_aug_512_BtoA_unity_0_100_cao_another_data_collection/train_10/images/fake_B'
]
real_dirs = ['/home/hao/hao/2025_june/CAO_virtuoso_data/unlabeled_data_preprocessed/image']
video_lists = ['mar28_model1_distance10_30',
             'apr29_model3_touch_1fps', 'apr29_model3_explore_1fps', 'apr29_model3_cut_1fps',
               'apr11_model1_1', 'apr11_model1_3',
               'mar30_model2_explore1', 'mar30_model2_explore2']

val_real_dir = '/home/hao/hao/2025_june/data/manual_mde_data'
for real_dir in real_dirs:
    for video_id in video_lists:
        image_list = sorted(os.listdir(os.path.join(real_dir, video_id)))
        for image in image_list:
            image_path = os.path.join(real_dir, video_id, image)
            split['train']['real'].append(image_path)

val_real_list = sorted(os.listdir(val_real_dir))
for val_real_video in val_real_list:
    image_dir = os.path.join(val_real_dir, val_real_video, 'image')
    depth_dir = os.path.join(val_real_dir, val_real_video, 'depth')

    image_list = sorted(os.listdir(image_dir))
    for image in image_list:
        image_path = os.path.join(image_dir, image)
        depth_path = os.path.join(depth_dir, image.replace('.jpg', '_depth.npy'))
        if os.path.exists(depth_path):
            data_pair = {'image': image_path, 'depth': depth_path}
            split['val_real'].append(data_pair)

for fakeB_dir in cut_dirs:
    if fakeB_dir == '/home/hao/hao/CUT_results/cut_baseline_aug_512_BtoA_unity_0_100_cao/train_10/images/fake_B':
        fabeBs = sorted(os.listdir(fakeB_dir))[::4]
    else:
        fabeBs = sorted(os.listdir(fakeB_dir))

    for fakeB_name in fabeBs:
        fakeB_path = os.path.join(fakeB_dir, fakeB_name)

        if fakeB_dir == '/home/hao/hao/CUT_results/cut_baseline_aug_512_BtoA_unity_0_100_cao/train_10/images/fake_B':
            model_name = fakeB_name.split('_render_')[0]
            image_id = fakeB_name.split('_render_')[1]
            depth_path = os.path.join(unity_data_dir, model_name, 'preprocessed_raw', image_id.replace('_render.png', '_dmap.npz'))
            #fakeB_path = os.path.join(unity_data_dir, model_name, 'render', image_id)

        else:
            model_name = fakeB_name.split('_p_render_')[0]
            image_id = fakeB_name.split('_p_render_')[1]
            depth_path = os.path.join(unity_data_dir1, model_name, 'p_raw', image_id.replace('_render.png', '_dmap.npy'))
            #fakeB_path = os.path.join(unity_data_dir1, model_name, 'p_render', image_id)

        if os.path.exists(depth_path):
            data_pair = {'image': fakeB_path, 'depth': depth_path}
            a = random.random()
            #if a < 0.2:
            #if 'cao_04_29_25_model4' in fakeB_name:
            if 'cao_03_20_25_model1' in fakeB_name or 'cao_03_20_25_model3' in fakeB_name or 'cao_04_29_25_model4' in fakeB_name:
                split['val_render'].append(data_pair)
            else:
                split['train']['render'].append(data_pair)

        else:
            print(depth_path)
            continue

output_path = 'spie_mde_json_0_100.json'
with open(output_path, 'w') as f:
    json.dump(split, f, indent=2)
