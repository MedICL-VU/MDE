import os
import shutil

image_folder = '/media/hao/mydrive1/CAO_virtuoso_data/CAO_images_segmentation-selected/Virtuoso_CAO_Dec_11_2024/images/WIN_20241211_16_51_43_Pro_modelAtumor2cut/'
image_list = sorted(os.listdir(image_folder))

for image in image_list:
    src = os.path.join(image_folder, image)
    new_image = image.replace('output_WIN_', '')

    dest = os.path.join(image_folder, new_image)
    shutil.move(src, dest)