import torch
import cv2
import lietorch
import droid_backends
import time
import argparse
import numpy as np
import open3d as o3d
import pathlib
from pathlib import Path

from lietorch import SE3
import geom.projective_ops as pops
import torch.nn.functional as F  # Added for interpolation

CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def create_camera_actor(g, scale=0.05):
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor

def create_point_actor(points, colors):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

def droid_visualization(video, device="cuda:0", args=None):
    torch.cuda.set_device(0)
    droid_visualization.video = video
    droid_visualization.cameras = {}
    droid_visualization.points = {}
    droid_visualization.warmup = 8
    droid_visualization.scale = 1.0
    droid_visualization.ix = 0
    print("droid_visualization")

    droid_visualization.filter_thresh = 0.15 # 0.3
    if not args is None:
        pointcloud_save_path = pathlib.Path(args.reconstruction_path)
    else:
        pointcloud_save_path = pathlib.path("./pc_outputs")
    pointcloud_save_path.mkdir(parents=True, exist_ok=True)
    print(pointcloud_save_path)

    def increase_filter(vis):
        droid_visualization.filter_thresh *= 2
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[:droid_visualization.video.counter.value] = True

    def decrease_filter(vis):
        droid_visualization.filter_thresh *= 0.5
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[:droid_visualization.video.counter.value] = True   

    def animation_callback(vis):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()

        with torch.no_grad():

            with video.get_lock():
                t = video.counter.value 
                dirty_index, = torch.where(video.dirty.clone())
                dirty_index = dirty_index

            if len(dirty_index) == 0:
                return
            video.dirty[dirty_index] = False

            poses = torch.index_select(video.poses, 0, dirty_index)
            disps = torch.index_select(video.disps, 0, dirty_index)
            Ps = SE3(poses).inv().matrix().cpu().numpy()

            images = torch.index_select(video.images, 0, dirty_index)
            images = images.cpu()[:,[2,1,0],3::8,3::8].permute(0,2,3,1) / 255.0
            points = droid_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics[0]).cpu()

            thresh = droid_visualization.filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))
            count = droid_backends.depth_filter(
                video.poses, video.disps, video.intrinsics[0], dirty_index, thresh)

            count = count.cpu()
            disps = disps.cpu()
            masks = ((count >= 2) & (disps > .5*disps.mean(dim=[1,2], keepdim=True)))

            if len(dirty_index) == t:
                print(f"All {t} frames needed visualization update")
                Path("{}/keyframe_data".format(pointcloud_save_path)).mkdir(parents=True, exist_ok=True)
                np.save("{}/keyframe_data/masks_from_vis.npy".format(pointcloud_save_path), masks.cpu().numpy())
                np.save("{}/keyframe_data/points_from_vis.npy".format(pointcloud_save_path), points.cpu().numpy())
                np.save("{}/keyframe_data/images_from_vis.npy".format(pointcloud_save_path), images.cpu().numpy())
                np.save("{}/keyframe_data/count_from_vis.npy".format(pointcloud_save_path), count.cpu().numpy())
                np.save("{}/keyframe_data/disps_from_vis.npy".format(pointcloud_save_path), disps.cpu().numpy())
                np.save("{}/keyframe_data/poses_from_vis.npy".format(pointcloud_save_path), poses.cpu().numpy())

            # count_up = droid_backends.depth_filter(
            #     video.poses, video.disps_up, video.intrinsics[0] * 8, dirty_index, thresh).cpu()
            # print(f"Count in visualization: {count_up.shape}, {count_up.numpy().mean()}")
            # print(f"poses in visualization: {video.poses.shape}, {video.poses.dtype}, {video.poses.cpu().numpy().mean()}")
            # print(f"disps_up in visualization: {video.disps_up.shape}, {video.disps_up.dtype}, {video.disps_up.cpu().numpy().mean()}")
            # print(f"intrinsics in visualization: {video.intrinsics.shape}, {video.intrinsics.dtype}, {video.intrinsics.cpu().numpy().mean()}")
            # print(f"disps in visualization: {video.disps.shape}, {video.disps.dtype}, {video.disps.cpu().numpy().mean()}")
            print(f"masks in visualization: {masks.shape}, {masks.dtype}, {masks.cpu().numpy().mean()}")
            # print(f"dirty in visualization: {dirty_index.shape}, {dirty_index.dtype}, {dirty_index.cpu().numpy().mean()}")

            all_segmented_pts, all_segmented_colors = [], []
            all_unsegmented_pts, all_unsegmented_colors = [], []

            for i in range(len(dirty_index)):
                pose = Ps[i]
                ix = dirty_index[i].item()

                if ix in droid_visualization.cameras:
                    vis.remove_geometry(droid_visualization.cameras[ix])
                    del droid_visualization.cameras[ix]

                if ix in droid_visualization.points:
                    vis.remove_geometry(droid_visualization.points[ix])
                    del droid_visualization.points[ix]

                cam_actor = create_camera_actor(True)
                cam_actor.transform(pose)
                vis.add_geometry(cam_actor)
                droid_visualization.cameras[ix] = cam_actor

                mask = masks[i].reshape(-1)
                pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
                clr = images[i].reshape(-1, 3)[mask].clone()

                if hasattr(video, 'masks'):
                    seg_mask = video.masks[ix]
                    H, W = disps[i].shape
                    seg_mask_resized = F.interpolate(seg_mask[None].float(), size=(H, W), mode="nearest")[0]
                    seg_mask = seg_mask_resized
                    seg_mask_flat = seg_mask.view(-1)[mask] > 0
                    # clr[seg_mask_flat] = torch.tensor([0.0, 0.0, 1.0], device=clr.device)

                    segmented_pts = pts[seg_mask_flat.cpu().numpy()]
                    segmented_colors = clr[seg_mask_flat.to(clr.device)].cpu().numpy()
                    unsegmented_pts = pts[~seg_mask_flat.cpu().numpy()]
                    unsegmented_colors = clr[(~seg_mask_flat).to(clr.device)].cpu().numpy()

                    all_segmented_pts.append(segmented_pts)
                    all_segmented_colors.append(segmented_colors)
                    all_unsegmented_pts.append(unsegmented_pts)
                    all_unsegmented_colors.append(unsegmented_colors)

                point_actor = create_point_actor(pts, clr.cpu().numpy())
                vis.add_geometry(point_actor)
                droid_visualization.points[ix] = point_actor


            all_segmented_pts_cat = np.concatenate(all_segmented_pts)
            all_segmented_colors_cat = np.concatenate(all_segmented_colors)
            if all_segmented_pts and all_segmented_pts_cat.shape[0] > 0:
                seg_pcd = o3d.geometry.PointCloud()
                seg_pcd.points = o3d.utility.Vector3dVector(all_segmented_pts_cat)
                seg_pcd.colors = o3d.utility.Vector3dVector(all_segmented_colors_cat)
                # print(f"saving tumor, {all_segmented_pts[0].shape}")
                output_path = f"{pointcloud_save_path}/target.ply"
                o3d.io.write_point_cloud(output_path, seg_pcd, write_ascii=False)

            all_unsegmented_pts_cat = np.concatenate(all_unsegmented_pts)
            all_unsegmented_colors_cat = np.concatenate(all_unsegmented_colors)
            if all_unsegmented_pts and all_unsegmented_pts_cat.shape[0] > 0:
                unseg_pcd = o3d.geometry.PointCloud()
                unseg_pcd.points = o3d.utility.Vector3dVector(all_unsegmented_pts_cat)
                unseg_pcd.colors = o3d.utility.Vector3dVector(all_unsegmented_colors_cat)
                # print(f"saving trachea, {all_unsegmented_pts[0].shape}")
                output_path = f"{pointcloud_save_path}/context.ply"
                o3d.io.write_point_cloud(output_path, unseg_pcd, write_ascii=False)

            pcd_camera = create_camera_actor(True)
            for c in droid_visualization.cameras.items():
                pcd_camera += c[1]
            
            output_path = f"{pointcloud_save_path}/camera.ply"
            o3d.io.write_line_set(output_path, pcd_camera, write_ascii=False)

            if len(droid_visualization.cameras) >= droid_visualization.warmup:
                cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam, True)

            droid_visualization.ix += 1
            vis.poll_events()
            vis.update_renderer()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_animation_callback(animation_callback)
    vis.register_key_callback(ord("S"), increase_filter)
    vis.register_key_callback(ord("A"), decrease_filter)

    vis.create_window(height=540, width=960)
    vis.get_render_option().load_from_json("misc/renderoption.json")

    vis.run()
    vis.destroy_window()
