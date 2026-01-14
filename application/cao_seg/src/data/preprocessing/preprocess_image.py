import glob
import cv2
import os

def load_crop_save(image_path, crop_coords, save_path, resize_dim=(224, 224)):
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print("Error: Unable to read image.")
        return

    # Convert to grayscale if needed
    if len(image.shape) == 2:
        pass  # Already grayscale
    elif len(image.shape) == 3:
        pass  # Keep as is (color)
    else:
        print("Error: Unsupported image format.")
        return

    # Crop
    x1, y1, x2, y2 = crop_coords
    cropped_frame = image[y1:y2, x1:x2] if len(image.shape) == 2 else image[y1:y2, x1:x2, :]

    # Resize
    resized_frame = cv2.resize(cropped_frame, resize_dim, interpolation=cv2.INTER_AREA)

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if '.jpg' in save_path:
        save_path = save_path.replace('.jpg', '.png')
    cv2.imwrite(save_path, resized_frame)
    print(f"Cropped + resized image saved at {save_path}")





base_dir = '/media/hao/mydrive1/CAO_virtuoso_data/unlabeled_data'
save_dir = '/media/hao/mydrive1/CAO_virtuoso_data/unlabeled_data_preprocessed'
labeled = False

image_paths = []
extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
for ext in extensions:
    image_paths.extend(glob.glob(os.path.join(base_dir, "image", "**", ext), recursive=True))

new_image_paths = []
label_paths = []

if labeled:
    for image_path in image_paths:
        label_path = image_path.replace('image/', 'mask/')
        label_path = label_path.replace('/frame_', '/defaultannot/frame_')
        if os.path.exists(label_path):
            label_paths.append(label_path)
            new_image_paths.append(image_path)
        else:
            print(label_path)
    assert len(new_image_paths) == len(label_paths)
else:
    new_image_paths = image_paths
    label_paths = image_paths



for image_path, label_path in zip(new_image_paths, label_paths):

    crop_coords = (320, 0, 1660, 1080)

    filename = image_path.split('/')[-1]

    input_path = image_path
    save_path = os.path.join(os.path.dirname(input_path), filename)
    save_path = save_path.replace(base_dir, save_dir)

    load_crop_save(input_path, crop_coords, save_path)
    if labeled:
        input_path = label_path
        save_path = os.path.join(os.path.dirname(input_path), filename)
        save_path = save_path.replace(base_dir, save_dir)
        load_crop_save(input_path, crop_coords, save_path)