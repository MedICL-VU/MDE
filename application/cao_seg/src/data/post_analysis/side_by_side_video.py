import cv2
import os
import glob
import numpy as np
from tqdm import tqdm
import time

def create_overlay_side_by_side_video(image_folder, segmentation_folder, output_video, fps=30, alpha=0.5, target_size=(512, 512)):
    """
    Creates a side-by-side video where:
    - Left: Cropped, resized original image (converted to RGB)
    - Right: Cropped, resized overlay of image + binary segmentation mask

    Parameters:
    - image_folder: Path to the folder containing original .npy images.
    - segmentation_folder: Path to the folder containing segmentation .npy masks.
    - output_video: Path to save the output video.
    - fps: Frames per second for the output video.
    - alpha: Transparency for overlay (0 = invisible, 1 = fully visible).
    - target_size: Resize images and segmentation masks to this size after cropping (default 512x512).
    """

    # Get sorted list of image and segmentation files
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.npy")))
    segmentation_paths = sorted(glob.glob(os.path.join(segmentation_folder, "*.png")))

    if len(image_paths) == 0 or len(segmentation_paths) == 0:
        print("Error: No .npy images or segmentation masks found.")
        return

    # Ensure both folders have the same number of frames
    if len(image_paths) != len(segmentation_paths):
        print("Warning: Image and segmentation count mismatch! Using the minimum count.")
        min_length = min(len(image_paths), len(segmentation_paths))
        image_paths = image_paths[:min_length]
        segmentation_paths = segmentation_paths[:min_length]

    # Define cropping dimensions
    crop_x1, crop_x2 = 320, 1660  # Crop width range
    crop_height = 1080

    # Video frame size: after cropping & resizing
    video_width = target_size[0] * 2  # Side-by-side
    video_height = target_size[1]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (video_width, video_height))

    print(f"Creating video: {output_video}")

    # Start timing
    start_time = time.time()

    # Progress bar
    for img_path, seg_path in tqdm(zip(image_paths, segmentation_paths), total=len(image_paths), desc="Processing Frames", unit="frame"):
        frame_start_time = time.time()

        # Load and crop original image
        orig_img = np.load(img_path, mmap_mode='r')
        if len(orig_img.shape) == 2:  # If grayscale, convert to RGB
            orig_img = np.stack([orig_img] * 3, axis=-1)

        orig_img = orig_img[:crop_height, crop_x1:crop_x2, :]  # Crop image (H, W, 3)

        # Convert BGR → RGB (Fixes color issue)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        # Load and crop segmentation mask
        #seg_mask = np.load(seg_path, mmap_mode='r')
        seg_mask = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        seg_mask = cv2.resize(seg_mask, (1920, 1080), interpolation=cv2.INTER_NEAREST)
        seg_mask = (seg_mask > 0).astype(np.uint8)  # Ensure binary mask
        seg_mask = seg_mask[:crop_height, crop_x1:crop_x2]  # Crop segmentation mask (H, W)

        # Resize both cropped images directly
        resize_start_time = time.time()
        orig_img_resized = cv2.resize(orig_img.astype(np.uint8), target_size, interpolation=cv2.INTER_LINEAR)
        seg_mask_resized = cv2.resize(seg_mask, target_size, interpolation=cv2.INTER_NEAREST)
        resize_time = time.time() - resize_start_time

        # Create a color mask for segmentation
        color_mask = np.zeros_like(orig_img_resized)
        color_mask[seg_mask_resized > 0] = [0, 0, 255]  # Red color for positive values

        # Apply weighted overlay
        overlayed_img = cv2.addWeighted(orig_img_resized, 1 - alpha, color_mask, alpha, 0)

        # Concatenate original and overlayed images side by side
        combined_frame = cv2.hconcat([orig_img_resized, overlayed_img])

        # Write frame directly to video (No extra memory usage)
        write_start_time = time.time()
        video_writer.write(combined_frame)
        write_time = time.time() - write_start_time

        frame_time = time.time() - frame_start_time

        # Print profiling info every 100 frames
        if img_path.endswith("000.npy"):
            print(f"[{img_path}] Resize: {resize_time:.4f}s | Write: {write_time:.4f}s | Total Frame: {frame_time:.4f}s")

    # Release video writer
    video_writer.release()
    total_time = time.time() - start_time
    print(f"Video saved at {output_video} | Total time: {total_time:.2f}s")



# Example usage
video_dir = "/media/hao/mydrive1/CAO_virtuoso_data/all_videos_frames"
curren_list = []

total_list = sorted(os.listdir(video_dir))
video_list = list(set(total_list) - set(curren_list))
# results_dir = video_dir.replace('_frames', '_predictions_mobileone_s1')
results_dir = video_dir.replace('_frames', '_predictions_mobileone_s1')

for video in video_list:
    image_dir = os.path.join(video_dir, video)
    segmentation_dir = os.path.join(results_dir, video)  # Folder containing binary segmentation masks
    output_video = f"{video}.mp4"  # Output video file path
    create_overlay_side_by_side_video(image_dir, segmentation_dir, output_video, fps=15, alpha=0.5)
