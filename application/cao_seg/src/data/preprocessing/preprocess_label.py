"""
get the largest connected component
"""

import cv2, glob, os
import numpy as np

def process_label(label_path, save_path):
    gray_image = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    unique_values = np.unique(gray_image)
    unique_values = unique_values[unique_values > 0]
    largest_area = 0
    largest_value = None
    for value in unique_values:
        # Create a binary mask for the current value
        mask = (gray_image == value).astype(np.uint8)

        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # Ignore background (label 0), find the largest connected component
        if num_labels > 1:
            max_area = max(stats[1:, cv2.CC_STAT_AREA])  # Get max area (excluding background)
            if max_area > largest_area:
                largest_area = max_area
                largest_value = value

    print(f"Largest connected component belongs to grayscale value: {largest_value} with area: {largest_area}")

    largest_image = (gray_image == largest_value).astype(np.uint8) * 255
    print(np.unique(largest_image))
    cv2.imwrite(save_path, largest_image)


# Example usage
base_dir = '/media/hao/mydrive1/CAO_virtuoso_data/labeled_data_preprocessed/mask'
save_dir = '/media/hao/mydrive1/CAO_virtuoso_data/labeled_data_preprocessed/mask_largest'
label_paths = []
extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
for ext in extensions:
    label_paths.extend(glob.glob(os.path.join(base_dir, "**", "*", ext), recursive=True))



for label_path in label_paths:
    print(f'processing {label_path}')
    save_path = label_path.replace(base_dir, save_dir)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    process_label(label_path, save_path)

