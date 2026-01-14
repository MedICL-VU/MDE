import numpy as np
import torch
import cv2
import os
import sys
from pathlib import Path
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.core.composition import Compose
from PIL import Image
import logging

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
import sys
# from models.segmentation_model import CATS2d
# from models.depth_anything_v2.dpt import DepthAnythingV2
from droid_slam.cao_seg.src.models.segmentation_model import CATS2d
from droid_slam.cao_seg.src.models.CATS2d_jane import CATS2d as CATS2d_2_channel
from droid_slam.cao_seg.src.models.depth_anything_v2.dpt import DepthAnythingV2
# from rembg import remove

class SegMDEInference:
    def __init__(self, cuda_device_id=0, run_seg=True, run_mde=True, seg_2_output_channels=False, checkpoint_path_seg=Path('droid_slam/cao_seg/src/models/spie_cao_tumor_segmentation.pth'), checkpoint_path_mde=Path('droid_slam/cao_seg/src/models/mde_cao_518.pth')):
        sys.path = list(dict.fromkeys(sys.path))
        if os.getcwd() not in sys.path:
            sys.path.append(os.getcwd())
        project_root = os.path.dirname(os.getcwd())
        if project_root in sys.path:
            sys.path.remove(project_root)

        if torch.cuda.is_available():
            self.DEVICE = torch.device(f"cuda:{cuda_device_id}")  # NVIDIA GPU
        elif torch.backends.mps.is_available():
            self.DEVICE = torch.device("cpu")  # Apple MPS (Mac M1/M2)
            #     DEVICE = torch.device("mps")  # Apple MPS (Mac M1/M2)
        else:
            self.DEVICE = torch.device("cpu")


        self.run_seg = run_seg
        self.run_mde = run_mde
        self.circle_mask = None
        self.seg_2_output_channels = seg_2_output_channels

        if self.run_seg:
            self._initialize_seg_model(checkpoint_path_seg=checkpoint_path_seg, seg_2_output_channels=seg_2_output_channels)
        
        if self.run_mde:
            self._initialize_mde_model(checkpoint_path_mde=checkpoint_path_mde)

        self.HD = False

        # pretrained model: https://vanderbilt.app.box.com/folder/309820360533
    
    def _initialize_seg_model(self, checkpoint_path_seg, seg_2_output_channels=False):
        if not self.run_seg:
            logging.warning("Warning: run_seg set to false, so seg_model is not initialized!")
            return
        self.TRANSFORM_INFERENCE_SEG = Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ], seed=42)
        self.CHECKPOINT_PATH_SEG = checkpoint_path_seg
        print(f"Loading Seg Model from {self.CHECKPOINT_PATH_SEG}")

        if not seg_2_output_channels:
            self.MODEL_SEG = CATS2d(checkpoint_path=None, device=self.DEVICE)
        else:
            self.MODEL_SEG = CATS2d_2_channel(checkpoint_path=None, device=self.DEVICE)

        self.MODEL_SEG.load_state_dict(torch.load(self.CHECKPOINT_PATH_SEG, map_location=self.DEVICE))
        self.MODEL_SEG.to(self.DEVICE)
        self.MODEL_SEG.eval()

    def _initialize_mde_model(self, checkpoint_path_mde):
        if not self.run_mde:
            logging.warning("run_mde set to false, so mde model is not initialized!")
            return
        
        self.TRANSFORM_INFERENCE_MDE = Compose([
            A.Resize(518, 518),
            # A.Resize(672, 672),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ], seed=42)
        self.CHECKPOINT_PATH_MDE = checkpoint_path_mde
        print(self.CHECKPOINT_PATH_MDE)
        MODEL_MDE_CONFIGS = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.MODEL_MDE = DepthAnythingV2(**{**MODEL_MDE_CONFIGS['vitb'], 'max_depth': 100}).to(self.DEVICE)
        self.MODEL_MDE.load_state_dict(torch.load(self.CHECKPOINT_PATH_MDE, map_location=self.DEVICE)['model'])
        self.MODEL_MDE = self.MODEL_MDE.to(self.DEVICE).eval()


    def create_circle_mask(self, frame_rgb, height=1080, width=1080, center=(540, 540), radius=475):
        # mask = np.zeros((height, width), dtype=np.uint8)
        # y, x = np.ogrid[:height, :width]
        # distance = (x - center[1]) ** 2 + (y - center[0]) ** 2
        # mask[distance <= radius ** 2] = 1
        
        # self.circle_mask = mask

        mask = np.ones((height, width), dtype=np.uint8)
        self.circle_mask = mask

        # binary_mask_path = 'binary_mask.png' # TODO: need to regenerate for each video?
        # if self.circle_mask is None:
        #     circle_mask = remove(frame_rgb, only_mask=True).copy()
        #     _, binary_mask = cv2.threshold(circle_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #     # cv2.imwrite(binary_mask_path, binary_mask)
        #     BINARY_MASK = (binary_mask / 255).astype(np.float32)
        #     self.circle_mask = BINARY_MASK
        # # else:
        #     # binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)
        # # BINARY_MASK = (binary_mask / 255).astype(np.float32)

        return self.circle_mask

    def resize_back(self, predict_depth, w=1920, h=1080):
        if self.HD:
            resized_array = cv2.resize(predict_depth, (1340, 1080), interpolation=cv2.INTER_NEAREST)
            pred_mask_resized = np.zeros((1080, 1920))
            pred_mask_resized[:, 320:1660] = resized_array
        else:
            # Resize mask to match the webcam feed size
            pred_mask_resized = cv2.resize(predict_depth, (w, h), interpolation=cv2.INTER_NEAREST)
            # pred_mask_resized = predict_depth
        return pred_mask_resized
    
    def inference_seg(self, frame_rgb: np.ndarray) -> np.ndarray:
        if not self.run_seg:
            logging.warning("run_seg set to false, so inference_seg not run!")
            return None
        logging.info("Seg Inference started...")
        self.MODEL_SEG.eval()

        self.HD = False
        h, w = frame_rgb.shape[:2]

        if w == 1920 and h == 1080:
            frame_rgb = frame_rgb[:, 320:1660, :]
            self.HD = True

        frame_rgb_transformed_seg = self.TRANSFORM_INFERENCE_SEG(image=frame_rgb)['image']
        frame_rgb_transformed_seg = np.moveaxis(frame_rgb_transformed_seg, -1, 0)
        frame_tensor_seg = torch.from_numpy(frame_rgb_transformed_seg).to(device=self.DEVICE, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            if type(self.MODEL_SEG).__name__ == 'CATS2d':
                output_seg = self.MODEL_SEG(frame_tensor_seg)[0]
            else:
                output_seg = self.MODEL_SEG(frame_tensor_seg)

            if self.seg_2_output_channels:
                output_seg_prob = F.softmax(output_seg, dim=1)[:, 1:2, ...]
            else:
                output_seg_prob = torch.sigmoid(output_seg)
            print(output_seg.shape)
            print(output_seg_prob.shape)
            pred_seg = (output_seg_prob > 0.5).float().squeeze().detach().cpu().numpy()
            pred_seg_resized = self.resize_back(pred_seg, w=w, h=h)
            circle_mask = self.create_circle_mask(frame_rgb=frame_rgb)

            masked_seg = pred_seg_resized * circle_mask
            logging.info("Seg Prediction done.")
            return masked_seg
        
    def inference_mde(self, frame_rgb: np.ndarray) -> np.ndarray:
        if not self.run_mde:
            logging.warning("run_mde set to false, so inference_mde not run!")
            return None
        logging.info("MDE Inference started...")
        self.MODEL_MDE.eval()

        self.HD = False
        h, w = frame_rgb.shape[:2]

        if w == 1920 and h == 1080:
            frame_rgb = frame_rgb[:, 320:1660, :]
            self.HD = True

        frame_rgb_transformed_mde = self.TRANSFORM_INFERENCE_MDE(image=frame_rgb)['image']
        frame_rgb_transformed_mde = np.moveaxis(frame_rgb_transformed_mde, -1, 0)
        frame_tensor_mde = torch.from_numpy(frame_rgb_transformed_mde).to(device=self.DEVICE, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_depth = self.MODEL_MDE(frame_tensor_mde)
            pred_depth = pred_depth.squeeze().detach().cpu().numpy()
            #pred_depth = 0.1 + pred_depth * (100 - 0.1) / 255

            pred_depth_resized = self.resize_back(pred_depth, w=w, h=h)
            #pred_depth_tumor = pred_depth_resized * pred_seg_resized
            circle_mask = self.create_circle_mask(frame_rgb=frame_rgb)

            masked_depth = pred_depth_resized * circle_mask
            logging.info("MDE prediction done.")

            return masked_depth        

    def inference_single(self, frame_rgb: np.ndarray) -> np.ndarray:
        print("Inference started...")
        self.MODEL_SEG.eval()
        self.MODEL_MDE.eval()

        self.HD = False
        h, w = frame_rgb.shape[:2]

        if w == 1920 and h == 1080:
            frame_rgb = frame_rgb[:, 320:1660, :]
            self.HD = True

        frame_rgb_transformed_seg = self.TRANSFORM_INFERENCE_SEG(image=frame_rgb)['image']
        frame_rgb_transformed_seg = np.moveaxis(frame_rgb_transformed_seg, -1, 0)
        frame_tensor_seg = torch.from_numpy(frame_rgb_transformed_seg).to(device=self.DEVICE, dtype=torch.float32).unsqueeze(0)

        frame_rgb_transformed_mde = self.TRANSFORM_INFERENCE_MDE(image=frame_rgb)['image']
        frame_rgb_transformed_mde = np.moveaxis(frame_rgb_transformed_mde, -1, 0)
        frame_tensor_mde = torch.from_numpy(frame_rgb_transformed_mde).to(device=self.DEVICE, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            if type(self.MODEL_SEG).__name__ == 'CATS2d':
                output_seg = self.MODEL_SEG(frame_tensor_seg)[0]
            else:
                output_seg = self.MODEL_SEG(frame_tensor_seg)


            if self.seg_2_output_channels:
                output_seg_prob = F.softmax(output_seg, dim=1)
            else:
                output_seg_prob = torch.sigmoid(output_seg)
            
            pred_seg = (output_seg_prob > 0.5).float().squeeze().detach().cpu().numpy()

            pred_depth = self.MODEL_MDE(frame_tensor_mde)
            pred_depth = pred_depth.squeeze().detach().cpu().numpy()
            #pred_depth = 0.1 + pred_depth * (100 - 0.1) / 255

            pred_depth_resized = self.resize_back(pred_depth, w=w, h=h)
            pred_seg_resized = self.resize_back(pred_seg, w=w, h=h)
            #pred_depth_tumor = pred_depth_resized * pred_seg_resized
            circle_mask = self.create_circle_mask(frame_rgb=frame_rgb)

            masked_seg = pred_seg_resized * circle_mask
            masked_depth = pred_depth_resized * circle_mask
            print("Prediction done.")

            return masked_seg, masked_depth
