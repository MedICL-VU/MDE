import cv2
import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mp
import os

def process_video_frame(image_path, crop_coords=(320, 0, 1660, 1080), threshold=20, selected_second=0):

    # Ensure output folder exists
    output_folder = os.getcwd()
    os.makedirs(output_folder, exist_ok=True)


    frame = cv2.imread(image_path) # Get first frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale

    # Crop the frame
    x1, y1, x2, y2 = crop_coords
    cropped_frame = frame_gray[y1:y2, x1:x2]

    # Resize cropped frame to 256x256
    resized_frame = cv2.resize(cropped_frame, (256, 256), interpolation=cv2.INTER_AREA)

    # Create binary mask from resized frame
    binary_mask = (resized_frame < threshold).astype(np.uint8)  # Pixels < threshold → 255 (white)




    binary_mask = np.ones_like(binary_mask) - binary_mask

    kernel = np.ones((10, 10), np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    binary_mask = cv2.erode(dilated_mask, kernel, iterations=1) * 255

    # Save individual images
    image_name = os.path.basename(image_path).split('.')[0]

    mask_path = os.path.join(output_folder, f"{image_name}_mask.png")
    subplot_path = os.path.join(output_folder, f"{image_name}_comparison.png")
    cv2.imwrite(mask_path, binary_mask)

    # Create subplot and save it
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(frame_gray, cmap='gray')
    ax[0].set_title("Original First Frame")
    ax[0].axis("off")

    ax[1].imshow(cropped_frame, cmap='gray')
    ax[1].set_title("Cropped Frame")
    ax[1].axis("off")

    ax[2].imshow(resized_frame, cmap='gray')
    ax[2].set_title("Resized (256x256)")
    ax[2].axis("off")

    ax[3].imshow(binary_mask, cmap='gray')
    ax[3].set_title("Binary Mask")
    ax[3].axis("off")

    plt.savefig(subplot_path, bbox_inches="tight")
    fig.suptitle(f"{image_path}")
    plt.show()

    print(f"Saved images in {output_folder}")

# Example Usage
image_path = "/media/hao/mydrive1/CAO_virtuoso_data/CAO_images_segmentation-selected/Virtuoso_CAO_Dec_11_2024/images/WIN_20241211_16_51_43_Pro_modelAtumor2cut/frame_00036.jpg"  # Replace with your actual video path
process_video_frame(image_path, selected_second=10, threshold=30)


# cropping coordinate for Dce. 2024 model crop_coords=(320, 0, 1460, 1080)