import os
import subprocess
from pathlib import Path
import glob


# Define the base directory where frames are stored
BASE_DIR = "/media/hao/mydrive1/CAO_virtuoso_data/all_videos_frames"
# Define the output directory
BASE_OUTPUT_DIR = "/media/hao/mydrive1/CAO_virtuoso_data/all_videos_predictions_mobileone_s1"
NAME = "CAO_cutoff_feb10_mobileone_s1"

def process_videos():
    """Finds all part_*.mp4 files in subdirectories and processes them with a Python command."""

    # Find all video files matching pattern "part_*.mp4"
    video_files = sorted(os.listdir(BASE_DIR))
    if not video_files:
        print("No matching video frames found.")
        return

    for video in video_files:
        frames_dir = os.path.join(BASE_DIR, video)
        save_results_dir = os.path.join(BASE_OUTPUT_DIR, video)
        print(f"Processing: {video}")
        # Construct the command
        cmd = [
            "python3", "/home/hao/Hao/CAO_seg/src/test.py",
            "--test_data_dir", str(frames_dir),
            "--name", str(NAME),
            "--save_results",
            "--save_results_dir", str(save_results_dir),
            "--height", str(256),
            "--width", str(256)
        ]
        print("Running command:", " ".join(cmd))  # Print the exact shell command

        # Run the command
        subprocess.run(cmd, check=True)  # Raises an error if the command fails

    print("Processing complete!")


# Run the processing function
if __name__ == "__main__":
    process_videos()
