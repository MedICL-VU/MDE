import os
import cv2
from genmsg import DURATION
from moviepy.editor import VideoFileClip, clips_array

def convert_video_to_gif(input_video, output_gif, fps=10):
    """
    Converts a video file to a GIF.

    Parameters:
    - input_video: Path to the input video file (e.g., "video.mp4")
    - output_gif: Path to save the output GIF file (e.g., "output.gif")
    - fps: Frames per second for the GIF (lower FPS = smaller file size)
    """
    clip = VideoFileClip(input_video)

    # Resize the GIF to reduce file size (optional)
    clip = clip.resize(0.5)  # 50% of the original size

    # Convert to GIF
    clip.write_gif(output_gif, fps=fps)
    print(f"✅ GIF saved as {output_gif}")

def extract_clip(input_video, start_time, duration, output_video):
    """Extracts a specific time segment from a video using OpenCV."""
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Cannot open {input_video}")
        return False

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_time * fps)
    end_frame = start_frame + int(duration * fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

    # Validate if the file was written correctly
    if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
        print(f"✅ Extracted: {output_video}")
        return True
    else:
        print(f"❌ Error: {output_video} was not created properly!")
        return False

DURATION_time=10
# Define video sources and timestamps
video_clips = [
    {"input": "dec11_model1_cut.mp4", "start_time": 0, "duration": DURATION_time, "output": "part1.mp4"},
    {"input": "dec11_model1_cut.mp4", "start_time": 24, "duration": DURATION_time, "output": "part2.mp4"},
    {"input": "feb10_model4_explore0.mp4", "start_time": 37, "duration": DURATION_time, "output": "part3.mp4"},
    {"input": "feb10_model4_light.mp4", "start_time": 24, "duration": DURATION_time, "output": "part4.mp4"},
    {"input": "feb11_model2_resection_retraction.mp4", "start_time": 4, "duration": DURATION_time, "output": "part5.mp4"},
    {"input": "feb11_model3_retraction2.mp4", "start_time": 7, "duration": DURATION_time, "output": "part6.mp4"}
]

# Extract video clips
for clip in video_clips:
    extract_clip(clip["input"], clip["start_time"], clip["duration"], clip["output"])

# Validate extracted clips before merging
valid_clips = []
for clip in video_clips:
    output_path = clip["output"]
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        try:
            valid_clips.append(VideoFileClip(output_path, fps_source="fps").resize((640, 360)))
        except Exception as e:
            print(f"⚠️ Skipping {output_path} due to error: {e}")

# Ensure there are valid clips before proceeding
if len(valid_clips) == 6:
    # Arrange videos in a 3x2 grid
    final_clip = clips_array([
        [valid_clips[0], valid_clips[1]],  # Row 1
        [valid_clips[2], valid_clips[3]],  # Row 2
        [valid_clips[4], valid_clips[5]]  # Row 3
    ])

    # Write the final combined video
    final_clip.write_videofile("part_all.mp4", codec="libx264", fps=15)
    print("🎬 Final 3x2 video created: part_all.mp4")
else:
    print("❌ Not enough valid clips to create a 3x2 video.")

from moviepy.editor import VideoFileClip


# Example Usage
convert_video_to_gif("part_all.mp4", "part_all.gif", fps=15)


