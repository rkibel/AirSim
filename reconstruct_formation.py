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
parser.add_argument('-s', '--step', default=1, type=int, help='frame step')
parser.add_argument('-t', '--depth_trunc', default=10000, type=float, help='max distance of depth projection')
parser.add_argument('--seg', action='store_true', help='use segmentation colors')
parser.add_argument('--vis', action='store_true', help='show visualization')
args = parser.parse_args()

# Get the default directory for AirSim
airsim_path = os.path.join(os.path.expanduser('~'), 'OneDrive', 'Documents', 'AirSim')

# Load camera settings from settings.json
with open(os.path.join(airsim_path, 'settings.json'), 'r') as fp:
    data = json.load(fp)
    
# Get camera settings (assuming all drones have same settings)
drone0_settings = data['Vehicles']['Drone0']['Cameras']['0']['CaptureSettings'][0]
img_width = drone0_settings['Width']
img_height = drone0_settings['Height']
img_fov = drone0_settings['FOV_Degrees']

# Create camera intrinsic object
fov_rad = img_fov * np.pi/180
fd = (img_width/2.0) / np.tan(fov_rad/2.0)
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    img_width, img_height, fd, fd, img_width/2.0, img_height/2.0
)

# Get the run name
if args.last:
    runs = []
    for f in os.listdir(airsim_path):
        if re.fullmatch('\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', f):
            runs.append(f)
    run = sorted(runs)[-1]
else:
    run = args.run

run_path = os.path.join(airsim_path, run)

def process_drone_data(data_path):
    """Process data for a single drone and return point cloud"""
    df = pd.read_csv(os.path.join(data_path, 'airsim_rec.txt'), delimiter='\t')
    pcd = o3d.geometry.PointCloud()
    
    for frame in tqdm(range(0, df.shape[0], args.step)):
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
        color_image = o3d.geometry.Image(np.asarray(color))
        depth_image = o3d.geometry.Image(depth)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image, depth_image, 
            depth_scale=1.0, 
            depth_trunc=args.depth_trunc, 
            convert_rgb_to_intensity=False
        )
        frame_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic, extrinsic=F
        )
        pcd += frame_pcd
    
    return pcd

# Process each drone and create combined point cloud
combined_pcd = o3d.geometry.PointCloud()
drone_pcds = {}

# Find all drone folders
drone_folders = [f for f in os.listdir(run_path) if f.startswith('Drone')]

print(f"Processing {len(drone_folders)} drones...")

# Process each drone
for drone in drone_folders:
    print(f"\nProcessing {drone}...")
    drone_path = os.path.join(run_path, drone)
    drone_pcd = process_drone_data(drone_path)
    
    # Save individual drone point cloud
    output_path = os.path.join(run_path, f'{drone}_pointcloud.ply')
    o3d.io.write_point_cloud(output_path, drone_pcd)
    print(f"Saved {drone} point cloud to {output_path}")
    
    # Add to combined point cloud
    combined_pcd += drone_pcd
    drone_pcds[drone] = drone_pcd

# Save combined point cloud
combined_output_path = os.path.join(run_path, 'combined_pointcloud.ply')
o3d.io.write_point_cloud(combined_output_path, combined_pcd)
print(f"\nSaved combined point cloud to {combined_output_path}")

# Visualize if requested
if args.vis:
    o3d.visualization.draw_geometries([combined_pcd])
