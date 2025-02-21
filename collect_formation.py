import time
import airsim
import numpy as np
import os
from datetime import datetime

# Set boundary parameters
wp_bound_x = [-100, 100]
wp_bound_y = [-100, 100]
wp_z = 50

# Formation parameters
N_CAMERAS = 4
FORMATION_ROWS = 2
FORMATION_COLS = 2
SPACING_X = 20
SPACING_Y = 20

# Flight parameters
SPEED = 5.0        # movement speed in m/s
DURATION = 35     # total flight duration in seconds
SAMPLE_TIME = 0.1  # time between pose updates

# Initialize AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.reset()
time.sleep(2)

# Initialize drones
vehicles = []
for i in range(N_CAMERAS):
    vehicle_name = f"Drone{i}"
    vehicles.append(vehicle_name)
    client.enableApiControl(True, vehicle_name)
    client.armDisarm(True, vehicle_name)

# Take off all drones
print("Taking off...")
tasks = []
for vehicle in vehicles:
    tasks.append(client.takeoffAsync(vehicle_name=vehicle))
for task in tasks:
    task.join()

# Generate formation offsets
formation_offsets = []
for row in range(FORMATION_ROWS):
    for col in range(FORMATION_COLS):
        offset_x = (col - (FORMATION_COLS-1)/2) * SPACING_X
        offset_y = (row - (FORMATION_ROWS-1)/2) * SPACING_Y
        formation_offsets.append((offset_x, offset_y))

# Create timestamp and directories for recording
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
base_path = os.path.join(os.path.expanduser('~'), 'OneDrive', 'Documents', 'AirSim', timestamp)

# Create directories for each drone
for vehicle in vehicles:
    drone_path = os.path.join(base_path, vehicle)
    os.makedirs(drone_path, exist_ok=True)

# Create airsim_rec.txt files for each drone
for vehicle in vehicles:
    drone_path = os.path.join(base_path, vehicle)
    with open(os.path.join(drone_path, 'airsim_rec.txt'), 'w') as f:
        f.write('VehicleName\tTimeStamp\tPOS_X\tPOS_Y\tPOS_Z\tQ_W\tQ_X\tQ_Y\tQ_Z\tImageFile\n')

# Initialize tracking variables
start_time = time.time()
last_positions = [(0, 0, wp_z) for _ in range(N_CAMERAS)]

try:
    while time.time() - start_time < DURATION:
        t = time.time() - start_time
        
        # Calculate base position (figure-8 pattern)
        scale_x = (wp_bound_x[1] - wp_bound_x[0]) / 2
        scale_y = (wp_bound_y[1] - wp_bound_y[0]) / 2
        center_x = (wp_bound_x[1] + wp_bound_x[0]) / 2
        center_y = (wp_bound_y[1] + wp_bound_y[0]) / 2
        
        # Slower, smoother figure-8
        x = center_x + scale_x * np.sin(2 * np.pi * t / 60)
        y = center_y + scale_y * np.sin(4 * np.pi * t / 60)
        z = wp_z
        
        # Move each drone in formation
        for i, (vehicle, offset) in enumerate(zip(vehicles, formation_offsets)):
            # Calculate drone position with smooth transition
            target_x = x + offset[0]
            target_y = y + offset[1]
            target_z = z
            
            # Smooth position transitions
            curr_x, curr_y, curr_z = last_positions[i]
            smooth_factor = 0.1  # lower = smoother movement
            
            drone_x = curr_x + (target_x - curr_x) * smooth_factor
            drone_y = curr_y + (target_y - curr_y) * smooth_factor
            drone_z = curr_z + (target_z - curr_z) * smooth_factor
            
            last_positions[i] = (drone_x, drone_y, drone_z)
            
            # Set the pose with fixed orientation (looking straight down)
            pose = airsim.Pose(
                airsim.Vector3r(drone_x, -drone_y, -drone_z),
                airsim.Quaternionr(0, 0, 0, 0) 
            )
            client.simSetVehiclePose(pose, True, vehicle_name=vehicle)
            
            # Capture images and record pose data
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene),
                airsim.ImageRequest("0", airsim.ImageType.DepthVis),
                airsim.ImageRequest("0", airsim.ImageType.Segmentation)
            ], vehicle_name=vehicle)
            
            # Save images and record data
            image_dir = os.path.join(base_path, vehicle, 'images')
            os.makedirs(image_dir, exist_ok=True)
            
            # Generate filenames and save images
            image_files = []
            for idx, response in enumerate(responses):
                if idx == 1:  # Depth image
                    filename = f'frame_{int(t*100):06d}_{idx}.pfm'
                    airsim.write_pfm(os.path.join(image_dir, filename), airsim.get_pfm_array(response))
                else:  # RGB and Segmentation images
                    filename = f'frame_{int(t*100):06d}_{idx}.png'
                    airsim.write_file(os.path.join(image_dir, filename), response.image_data_uint8)
                image_files.append(filename)
            
            # Record pose data
            with open(os.path.join(base_path, vehicle, 'airsim_rec.txt'), 'a') as f:
                q = pose.orientation
                image_names = ';'.join(image_files)
                f.write(f'{vehicle}\t{int(t*1e9)}\t{drone_x}\t{drone_y}\t{drone_z}\t{q.w_val}\t{q.x_val}\t{q.y_val}\t{q.z_val}\t{image_names}\n')
        
        time.sleep(SAMPLE_TIME)

finally:
    # Land all drones
    print("Landing...")
    tasks = []
    for vehicle in vehicles:
        tasks.append(client.landAsync(vehicle_name=vehicle))
    for task in tasks:
        task.join()
    
    # Disarm and release control
    for vehicle in vehicles:
        client.armDisarm(False, vehicle_name=vehicle)
        client.enableApiControl(False, vehicle_name=vehicle)
