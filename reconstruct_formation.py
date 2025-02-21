import argparse
import json
import os
import re

import airsim
import numpy as np
import open3d as o3d
import pandas as pd
import PIL.Image
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-r', '--run', help='folder name of the run (timestamp)')
group.add_argument('-l', '--last', action='store_true', help='use last run')
parser.add_argument('-d', '--drone', help='specific drone to reconstruct (e.g., Drone0)', default=None)
parser.add_argument('-c', '--combine', action='store_true', help='combine all drone data into single point cloud')
parser.add_argument('-s', '--step', default=1, type=int, help='frame step')
parser.add_argument('-t', '--depth_trunc', default=10000, type=float, help='max distance of depth projection')
parser.add_argument('-w', '--write_frames', action='store_true', help='save a point cloud for each frame')
parser.add_argument('--seg', action='store_true', help='use segmentation colors')
parser.add_argument('--vis', action='store_true', help='show visualization')
args = parser.parse_args()

# Get the default directory for AirSim and load settings
airsim_path = os.path.join(os.path.expanduser('~'), 'OneDrive', 'Documents', 'AirSim')

# Load camera settings
with open(os.path.join(airsim_path, 'settings.json'), 'r') as fp:
    data = json.load(fp)
capture_settings = data['CameraDefaults']['CaptureSettings'][0]
img_width = capture_settings['Width']
img_height = capture_settings['Height']
img_fov = capture_settings['FOV_Degrees']

# Create camera intrinsic object
fov_rad = img_fov * np.pi/180
fd = (img_width/2.0) / np.tan(fov_rad/2.0)
intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(img_width, img_height, fd, fd, img_width/2 - 0.5, img_height/2 - 0.5)

# Get the run name
if args.last:
    runs = []
    for f in os.listdir(airsim_path):
        if re.fullmatch('\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', f):
            runs.append(f)
    run = sorted(runs)[-1]
else:
    run = args.run

# Environment bounds
ENVIRONMENT_BOUNDS = {
    'x': (-150, 150),
    'y': (-150, 150),
    'z': (0, 150)
}

def process_drone_data(data_path, step, write_frames):
    """Process data for a single drone and return point cloud and cameras"""
    df = pd.read_csv(os.path.join(data_path, 'airsim_rec.txt'), delimiter='\t')
    pcd = o3d.geometry.PointCloud()
    cams = []
    
    if write_frames:
        os.makedirs(os.path.join(data_path, 'points'), exist_ok=True)
    
    for frame in tqdm(range(0, df.shape[0], step)):
        # Create transformation matrix
        x, y, z = df.iloc[frame][['POS_X', 'POS_Y', 'POS_Z']]
        qw, qx, qy, qz = df.iloc[frame][['Q_W', 'Q_X', 'Q_Y', 'Q_Z']]
        
        T = np.eye(4)
        T[:3,3] = [-y, -z, -x]
        R = np.eye(4)
        R[:3,:3] = o3d.geometry.get_rotation_matrix_from_quaternion((qw, qy, qz, qx))
        C = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        F = R.T @ T @ C

        # Load and process images
        rgb_filename, seg_filename, depth_filename = df.iloc[frame].ImageFile.split(';')
        rgb = PIL.Image.open(os.path.join(data_path, 'images', rgb_filename)).convert('RGB')
        seg = PIL.Image.open(os.path.join(data_path, 'images', seg_filename)).convert('RGB')
        depth, _ = airsim.utils.read_pfm(os.path.join(data_path, 'images', depth_filename))

        # Create point cloud
        color = seg if args.seg else rgb
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.asarray(color)),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=args.depth_trunc,
            convert_rgb_to_intensity=False
        )
        rgbd_pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, extrinsic=F)

        # Filter points
        points = np.asarray(rgbd_pc.points)
        colors = np.asarray(rgbd_pc.colors)
        mask = (points[:, 0] >= ENVIRONMENT_BOUNDS['x'][0]) & (points[:, 0] <= ENVIRONMENT_BOUNDS['x'][1]) & \
               (points[:, 1] >= ENVIRONMENT_BOUNDS['y'][0]) & (points[:, 1] <= ENVIRONMENT_BOUNDS['y'][1]) & \
               (points[:, 2] >= ENVIRONMENT_BOUNDS['z'][0]) & (points[:, 2] <= ENVIRONMENT_BOUNDS['z'][1])
        
        rgbd_pc.points = o3d.utility.Vector3dVector(points[mask])
        rgbd_pc.colors = o3d.utility.Vector3dVector(colors[mask])
        pcd += rgbd_pc

        if write_frames:
            pcd_name = f'points_seg_{frame:06d}' if args.seg else f'points_rgb_{frame:06d}'
            o3d.io.write_point_cloud(os.path.join(data_path, 'points', f'{pcd_name}.pcd'), rgbd_pc)
            
            cam = o3d.camera.PinholeCameraParameters()
            cam.intrinsic = intrinsic
            cam.extrinsic = F
            o3d.io.write_pinhole_camera_parameters(
                os.path.join(data_path, 'points', f'cam_{frame:06d}.json'), cam)

        cams.append(o3d.geometry.LineSet.create_camera_visualization(intrinsic, F))
    
    return pcd, cams

# Process drones based on arguments
if args.drone:
    # Single drone reconstruction
    data_path = os.path.join(airsim_path, run, args.drone)
    pcd, cams = process_drone_data(data_path, args.step, args.write_frames)
elif args.combine:
    # Combine all drones
    pcd = o3d.geometry.PointCloud()
    cams = []
    for drone_dir in os.listdir(os.path.join(airsim_path, run)):
        if drone_dir.startswith('Drone'):
            data_path = os.path.join(airsim_path, run, drone_dir)
            drone_pcd, drone_cams = process_drone_data(data_path, args.step, args.write_frames)
            pcd += drone_pcd
            cams.extend(drone_cams)

# Save the final point cloud
pcd_name = 'points_seg' if args.seg else 'points_rgb'
pcd_path = os.path.join(data_path, pcd_name + '.pcd')
o3d.io.write_point_cloud(pcd_path, pcd)

# Visualize if requested
if args.vis:
    o3d.visualization.draw_geometries([pcd] + cams)
