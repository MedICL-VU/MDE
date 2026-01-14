import os
import json, glob
import random

base_dir = '/media/hao/mydrive1/CAO_virtuoso_data/cropped_data'
image_paths = glob.glob(base_dir + '/**/images/**/*/*.png', recursive=True)
label_paths = glob.glob(base_dir + '/**/GT_masks/**/*/*.png', recursive=True)

assert len(image_paths) == len(label_paths)

split = {'train':[], 'val': []}


for image_path, label_path in zip(image_paths, label_paths):
    image_to_label = image_path.replace('/images', '/GT_masks').replace('/frame', '/defaultannot/frame')
    assert image_to_label == label_path

    paird_data = {'image': image_path, 'label': label_path}
    if random.random() < 0.8:
        split['train'].append(paird_data)
    else:
        split['val'].append(paird_data)
with open('split.json', 'w') as f:
    json.dump(split, f, indent=4)
print(1)