import cv2
import os
import glob


def extract_frames_to_png(video_folder, output_folder, frame_interval=1):
    """
    Extracts frames from all videos in the folder and saves them as .png files with timestamps.

    Parameters:
    - video_folder: Path to the folder containing videos.
    - output_folder: Path to save extracted frames as .png files.
    - frame_interval: Save every Nth frame (default is 1, meaning every frame).
    """

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all video files (supports mp4, avi, mov, mkv, etc.)
    video_paths = glob.glob(os.path.join(video_folder, "*.*"))
    video_paths = [v for v in video_paths if v.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not video_paths:
        print("No videos found in the folder.")
        return

    for video_path in video_paths:
        # Extract video name (without extension) for subfolder
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_folder = os.path.join(output_folder, video_name)

        # Create a subfolder for this video's frames
        os.makedirs(video_output_folder, exist_ok=True)

        print(f"Processing: {video_name}")

        # Open video file
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break  # Break when video ends

            if frame_count % frame_interval == 0:
                # Compute timestamp
                milliseconds = int((frame_count / fps) * 1000)
                minutes = (milliseconds // 60000) % 60
                seconds = (milliseconds // 1000) % 60
                millis = milliseconds % 1000

                frame_filename = os.path.join(video_output_folder, f"frame_{minutes:02d}{seconds:02d}{millis:03d}.png")

                # Save frame as PNG
                #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(frame_filename, frame)

            frame_count += 1

        cap.release()
        print(f"Extracted frames from {video_name}.")

    print("Frame extraction and saving completed.")


# Example usage
video_folder = "/media/hao/mydrive1/CAO_virtuoso_data/all_videos/1"  # Change to your folder with videos
output_folder = "/media/hao/mydrive1/CAO_virtuoso_data/unlabeled_data/image"  # Where frames will be saved


extract_frames_to_png(video_folder, output_folder, frame_interval=10)  # Extract every 10th frame
