import os
import cv2
import numpy as np
import glob

# Input and output directories
input_folder = "/media/hao/mydrive1/download/Segmentation_Test/Dec11_ModelA/Tumor1/segs_256"  # Change to your folder
output_folder = "/media/hao/mydrive1/download/Segmentation_Test/Dec11_ModelA/Tumor1/pseudo_masks_hard"  # Change to your output folder

# input_folder = "/media/hao/mydrive1/download/Segmentation_Test/Feb11_Model3/Exploration_1/segs_256"  # Change to your folder
# output_folder = "/media/hao/mydrive1/download/Segmentation_Test/Feb11_Model3/Exploration_1/pseudo_masks_hard"  # Change to your output folder

# input_folder = '/media/hao/mydrive1/download/Segmentation_Test/Feb11_Model1/sequence1/segs_256'
# output_folder = '/media/hao/mydrive1/download/Segmentation_Test/Feb11_Model1/sequence1/pseudo_masks_hard'


# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Define the morphological kernel (31x31)
kernel = np.ones((101, 101), np.uint8)

# Get all mask files
mask_files = glob.glob(os.path.join(input_folder, "*.png"))  # Change extension if needed

for mask_path in mask_files:
    # Load binary mask (grayscale)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure binary values (255 foreground, 0 background)
    mask[mask > 0] = 255

    # Apply morphological operations
    mask = cv2.erode(mask, kernel, iterations=1)   # Erosion to remove noise
    mask = cv2.dilate(mask, kernel, iterations=1)  # Dilation to restore structure

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:  # No valid components (only background)
        processed_mask = np.zeros_like(mask)  # Blank mask
    else:
        # Get component areas (excluding background, index 0)
        areas = stats[1:, cv2.CC_STAT_AREA]  # Exclude background
        sorted_indices = np.argsort(areas)[::-1]  # Sort in descending order (largest first)

        # Always check the largest component first
        largest_label = sorted_indices[0] + 1  # Adjust index to match labels
        largest_area = stats[largest_label, cv2.CC_STAT_AREA]

        if largest_area < 2000:
            processed_mask = np.zeros_like(mask)  # Make blank if largest < 2000
        else:
            processed_mask = np.zeros_like(mask)
            processed_mask[labels == largest_label] = 255  # Keep the largest component

            # Check second largest component condition
            if len(sorted_indices) > 1:
                second_largest_label = sorted_indices[1] + 1
                second_largest_area = stats[second_largest_label, cv2.CC_STAT_AREA]

                if second_largest_area >= largest_area / 10:
                    processed_mask[labels == second_largest_label] = 255  # Keep second component

    # Save the processed mask
    output_path = os.path.join(output_folder, os.path.basename(mask_path))
    cv2.imwrite(output_path, processed_mask)
    print(f"Processed and saved: {output_path}")

print("✅ Processing complete.")