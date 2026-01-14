import numpy as np
import matplotlib.pyplot as plt


def overlay_and_plot(image_path, label_path, alpha=0.5):
    """
    Loads an image and a label from .npy files, overlays them, and plots the result.

    Parameters:
    - image_path: Path to the .npy image file
    - label_path: Path to the .npy label file
    - alpha: Transparency for the overlay (0 = invisible, 1 = fully visible)
    """

    # Load the image and label
    image = np.load(image_path)
    label = np.load(label_path)

    # Ensure the image is RGB (if grayscale, convert it)
    if len(image.shape) == 2:  # Grayscale image
        image = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB

    # Ensure the label is a 2D array
    if len(label.shape) == 3 and label.shape[-1] == 3:
        label = np.mean(label, axis=-1)  # Convert RGB label to grayscale

    # Normalize label values for visualization
    label = (label - label.min()) / (label.max() - label.min())  # Scale between 0 and 1

    # Define colormap for label overlay
    cmap = plt.cm.jet  # Change to any colormap you like (e.g., `plt.cm.viridis`)

    # Convert label to an RGBA heatmap
    label_colored = cmap(label)[:, :, :3]  # Get RGB from colormap (ignore alpha channel)

    # Blend the image and the label overlay
    overlay = (1 - alpha) * image + alpha * label_colored * 255  # Scale label to match image intensity

    # Plot the original image, label, and overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image.astype(np.uint8))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(label, cmap="gray")
    axes[1].set_title("Label Mask")
    axes[1].axis("off")

    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.show()


# Example usage
image_npy = "/media/hao/mydrive1/CAO_virtuoso_data/all_videos_frames/mar05_jhu_2_2/frame_00000000.npy"  # Replace with actual path
label_npy = "/media/hao/mydrive1/CAO_virtuoso_data/all_videos_predictions/mar05_jhu_2_2/frame_00000000.npy"  # Replace with actual path
overlay_and_plot(image_npy, label_npy, alpha=0.5)
