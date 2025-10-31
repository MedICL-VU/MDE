import os
import cv2
import numpy as np
from glob import glob
from pathlib import Path
from typing import List, Tuple

# ----------------------------
# EDIT THESE
image_dir = "/home/hao/hao/mde_ct_camera/mcap_to_vision/images_cao"   # contains .png / .jpg / .jpeg
depth_dir = "/home/hao/hao/mde_ct_camera/mcap_to_vision/depths_cao"   # contains .npy (HxW or HxWx1)
out_path  = "/home/hao/hao/mde_ct_camera/mcap_to_vision/cao_output_side_by_side.mp4"
fps       = 20
size      = (518, 518)                 # (width, height)
# Depth normalization is FIXED to 0..100 mm (per your request)
depth_min_mm = 0.0
depth_max_mm = 100.0
# If your .npy values are not in millimeters, set a scale (e.g., meters->mm: 1000.0)
depth_scale_to_mm = 1.0
# Colorbar appearance
cbar_width_px = 60
tick_mm = [0, 25, 50, 75, 100]
colormap = cv2.COLORMAP_INFERNO  # change if you prefer
# ----------------------------

def list_images(folder: str) -> List[Path]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    files = []
    for e in exts:
        files.extend(glob(os.path.join(folder, e)))
    return sorted(map(Path, files))

def list_depths(folder: str) -> List[Path]:
    return sorted(map(Path, glob(os.path.join(folder, "*.npy"))))

def load_depth(path: Path) -> np.ndarray:
    d = np.load(str(path))
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    elif d.ndim != 2:
        raise ValueError(f"Depth array must be HxW or HxWx1. Got shape {d.shape} from {path.name}")
    return d.astype(np.float32)

def to_bgr(im: np.ndarray) -> np.ndarray:
    if im.ndim == 2:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 3:
        return im
    if im.shape[2] == 4:
        return cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Unexpected image shape: {im.shape}")

def resize_square(im: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(im, size_wh, interpolation=cv2.INTER_AREA)

def match_by_stem(img_paths: List[Path], dpt_paths: List[Path]) -> List[Tuple[Path, Path]]:
    imgs_by_stem = {p.stem: p for p in img_paths}
    dpts_by_stem = {p.stem: p for p in dpt_paths}
    common = sorted(set(imgs_by_stem.keys()) & set(dpts_by_stem.keys()))
    pairs = [(imgs_by_stem[s], dpts_by_stem[s]) for s in common]
    if not pairs:
        n = min(len(img_paths), len(dpt_paths))
        pairs = list(zip(img_paths[:n], dpt_paths[:n]))
    return pairs

def colorize_depth_mm(depth_mm: np.ndarray, lo_mm: float, hi_mm: float) -> np.ndarray:
    """Map depth in mm to [0,255], apply OpenCV colormap."""
    d = np.clip(depth_mm, lo_mm, hi_mm)
    norm = (d - lo_mm) / (hi_mm - lo_mm + 1e-12)
    img8 = np.uint8(np.round(norm * 255.0))
    colored = cv2.applyColorMap(img8, colormap)  # HxWx3 BGR
    return colored

def make_vertical_colorbar(height: int,
                           width: int,
                           lo_mm: float,
                           hi_mm: float,
                           ticks_mm: List[float],
                           label: str = "mm") -> np.ndarray:
    """
    Create a vertical colorbar image (HxWx3 BGR) with given height/width.
    Top = hi_mm, Bottom = lo_mm, matching the depth colormap mapping.
    """
    # Create vertical gradient (top 255 -> bottom 0)
    grad = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)
    grad = np.repeat(grad, width, axis=1)  # HxW
    cbar = cv2.applyColorMap(grad, colormap)

    # Add ticks & labels
    # Map tick (mm) -> y pixel (0 at top; height-1 at bottom)
    def mm_to_y(mm):
        frac = (mm - lo_mm) / (hi_mm - lo_mm + 1e-12)
        y = (1.0 - frac) * (height - 1)  # invert: hi at top
        return int(np.clip(np.round(y), 0, height - 1))

    # draw tick marks
    for mm in ticks_mm:
        y = mm_to_y(mm)
        cv2.line(cbar, (0, y), (int(width * 0.35), y), (0, 0, 0), 2)          # small black tick
        cv2.line(cbar, (int(width * 0.35), y), (int(width * 0.4), y), (255,255,255), 1)  # white edge

        text = f"{int(mm)}"
        # Put text slightly right of the ticks
        cv2.putText(cbar, text, (int(width * 0.45), y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(cbar, text, (int(width * 0.45), y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    # Unit label at top
    cv2.putText(cbar, label, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(cbar, label, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

    return cbar

def main():
    img_paths = list_images(image_dir)[-500:-1]
    dpt_paths = list_depths(depth_dir)[-500:-1]

    if not img_paths:
        raise FileNotFoundError(f"No PNG/JPG images found in {image_dir}")
    if not dpt_paths:
        raise FileNotFoundError(f"No .npy depth files found in {depth_dir}")

    pairs = match_by_stem(img_paths, dpt_paths)
    print(f"Frames to write: {len(pairs)}")

    w, h = size
    frame_size = (w * 2 + cbar_width_px, h)  # image | depth | colorbar
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, frame_size, True)
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open writer for {out_path}")

    # Pre-build colorbar at target height so it's consistent per-frame
    cbar = make_vertical_colorbar(
        height=h, width=cbar_width_px,
        lo_mm=depth_min_mm, hi_mm=depth_max_mm,
        ticks_mm=tick_mm, label="mm"
    )

    for i, (ip, dp) in enumerate(pairs):
        # Image
        img_bgr = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            print(f"[WARN] Skipping unreadable image: {ip}")
            continue
        img_bgr = to_bgr(img_bgr)
        img_bgr = resize_square(img_bgr, size)

        # Depth -> mm -> colorize with fixed 0..100 mm
        d_raw = load_depth(dp)  # HxW (float32)
        d_mm = d_raw * float(depth_scale_to_mm)
        d_mm_resized = cv2.resize(d_mm, size, interpolation=cv2.INTER_NEAREST)
        depth_bgr = colorize_depth_mm(d_mm_resized, depth_min_mm, depth_max_mm)

        # Concatenate: [image | depth | colorbar]
        frame = np.hstack([img_bgr, depth_bgr, cbar])

        # Optional index overlay
        cv2.putText(frame, f"{i:06d}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        vw.write(frame)
        if (i+1) % 50 == 0:
            print(f"Wrote {i+1} frames")

    vw.release()
    print(f"Done. Saved to: {out_path}")

if __name__ == "__main__":
    main()