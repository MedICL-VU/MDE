import os.path

from dataset.CAO_dataset import CAO_dataset
from torch.utils.data import DataLoader
from albumentations.core.composition import Compose
from albumentations import Resize
import numpy as np
import torch
from tqdm import tqdm
from util.utils import dice_coefficient_multiclass_batch
from config.config_args import *
from config.config_setup import get_net, init_seeds
from PIL import Image
from config.config_setup import get_dataset
import cv2
import os
import sys
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.core.composition import Compose
from models.CATS2d import CATS2d
import time

sys.path = list(dict.fromkeys(sys.path))
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
project_root = os.path.dirname(os.getcwd())
if project_root in sys.path:
    sys.path.remove(project_root)

if torch.cuda.is_available():
    DEVICE = torch.device(f"cuda:{0}")  # NVIDIA GPU
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Apple MPS (Mac M1/M2)
else:
    DEVICE = torch.device("cpu")

TRANSFORM_INFERENCE = Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
], seed=42)

HD = False


def inference(net1):
    net1.eval()
    cap = cv2.VideoCapture(0)

    while True:
        start_time = time.time()  # Start timing
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print('Image not captured')
            break

        #with torch.no_grad():
        with torch.inference_mode():
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]


            if w == 1920 and h == 1080:
                frame_rgb = frame_rgb[:, 320:1660, :]
                HD = True

            frame_rgb_transformed = TRANSFORM_INFERENCE(image=frame_rgb)['image']
            frame_rgb_transformed = np.moveaxis(frame_rgb_transformed, -1, 0)
            frame_tensor = torch.from_numpy(frame_rgb_transformed).to(device=DEVICE, dtype=torch.float32).unsqueeze(0)

            output_prob = torch.sigmoid(net(frame_tensor))
            pred_mask = (output_prob > 0.5).float().squeeze().detach().cpu().numpy()

            if HD:
                resized_array = cv2.resize(pred_mask, (1340, 1080), interpolation=cv2.INTER_NEAREST)
                pred_mask_resized = np.zeros((1080, 1920))
                pred_mask_resized[:, 320:1660] = resized_array
            else:
                # Resize mask to match the webcam feed size
                pred_mask_resized = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)


            """visualization below"""
            # Convert to 3-channel binary mask (for better visualization)
            mask_colored = np.zeros_like(frame)  # Create an empty frame with the same size
            mask_colored[:, :, 2] = pred_mask_resized * 255  # Set only the Red channel
            mask_colored = mask_colored.astype(np.uint8)

            # Overlay mask on the original frame
            overlay = cv2.addWeighted(frame, 0.6, mask_colored, 0.4, 0)

            # Display the overlay window
            cv2.imshow("Segmentation Output", overlay)

            end_time = time.time()  # End timing
            processing_time = end_time - start_time  # Calculate frame processing time

            print(f"Processing time per frame with shape {(h,w)}: {processing_time:.3f} seconds ({1/processing_time:.2f} FPS)")

            # Press ESC to exit
            if cv2.waitKey(1) & 0xFF == 27:
                break



    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    checkpoint_path = os.path.dirname(__file__) + '/best_net.pth'
    inference_mode = True
    net = smp.Unet(encoder_name='mobileone_s1', encoder_weights=None, in_channels=3, classes=1)
    net.encoder.inference_mode = inference_mode


    model_state = torch.load(checkpoint_path, map_location=DEVICE)
    net.load_state_dict(model_state)
    net.to(DEVICE)

    inference(net1=net)



    # if args.save_results:
    #     frame_name = batch['name'][0].split('/')[-1]
    #     predict_mask_numpy = pred_labels.cpu().numpy().squeeze()
    #     resized_array = cv2.resize(predict_mask_numpy, (1340, 1080), interpolation=cv2.INTER_NEAREST)
    #
    #     blank_mask = np.zeros((1080, 1920))
    #     blank_mask[:, 320:1660] = resized_array
    #
    #     blank_mask_resized = cv2.resize(blank_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    #     blank_mask_resized = blank_mask_resized.astype(np.uint8)
    #     cv2.imwrite(os.path.join(save_results_dir, frame_name.replace('.npy', '.png')), blank_mask_resized)
    #                     np.save(os.path.join(save_results_dir, frame_name), blank_mask)
