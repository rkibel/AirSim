import time
import airsim
import numpy as np
import os
from datetime import datetime

# Set boundary parameters
wp_bound_x = [-100, 100]
wp_bound_y = [-100, 100]
wp_z = 50

# Flight parameters
DURATION = 32
TIME_INCREMENT = 0.0008

# Camera parameters
PITCH_RANGE = [-90, -45]  # degrees, -90 is down, -45 is forward-down
ROLL_RANGE = [-20, 20]    # degrees, for banking turns
YAW_RATE = 1.0           # reduced for smoother rotation

VEHICLES = ["Drone0", "Drone1", "Drone2", "Drone3"]
FORMATION_SPACING = 8.0  
DRONE_POSITIONS = [
    [-FORMATION_SPACING/2, -FORMATION_SPACING/2],     
    [FORMATION_SPACING/2, -FORMATION_SPACING/2],      
    [-FORMATION_SPACING/2, FORMATION_SPACING/2],      
    [FORMATION_SPACING/2, FORMATION_SPACING/2]        
]

# Connect to AirSim
client = airsim.MultirotorClient()
client.reset()
client.confirmConnection()

for drone_name in VEHICLES:
    client.enableApiControl(True, drone_name)
    client.armDisarm(True, drone_name)

# Create output directory with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
base_path = os.path.join(os.path.expanduser('~'), 'OneDrive', 'Documents', 'AirSim', timestamp)
os.makedirs(base_path, exist_ok=True)

# Create directories for each drone
for vehicle in VEHICLES:
    drone_path = os.path.join(base_path, vehicle)
    os.makedirs(drone_path, exist_ok=True)

# Create airsim_rec.txt files for each drone
for vehicle in VEHICLES:
    drone_path = os.path.join(base_path, vehicle)
    image_path = os.path.join(drone_path, 'images')
    os.makedirs(image_path, exist_ok=True)
    with open(os.path.join(drone_path, 'airsim_rec.txt'), 'w') as f:
        f.write('VehicleName\tTimeStamp\tPOS_X\tPOS_Y\tPOS_Z\tQ_W\tQ_X\tQ_Y\tQ_Z\tImageFile\n')

time.sleep(1)

t = 0
last_yaw = 0
frame = 0

try:
    while t < DURATION:
        # Generate smooth figure-8 pattern for formation center
        scale_x = (wp_bound_x[1] - wp_bound_x[0]) / 2
        scale_y = (wp_bound_y[1] - wp_bound_y[0]) / 2
        center_x = (wp_bound_x[1] + wp_bound_x[0]) / 2
        center_y = (wp_bound_y[1] + wp_bound_y[0]) / 2
        
        # Position equations for formation center
        formation_x = center_x + scale_x * np.sin(2 * np.pi * t / 30)
        formation_y = center_y + scale_y * np.sin(4 * np.pi * t / 30)
        
        # Calculate yaw with continuous rotation
        dx = scale_x * (2 * np.pi / 30) * np.cos(2 * np.pi * t / 30)
        dy = scale_y * (4 * np.pi / 30) * np.cos(4 * np.pi * t / 30)
        target_yaw = np.arctan2(dy, dx)
        
        # Ensure smooth yaw transitions
        yaw_diff = np.arctan2(np.sin(target_yaw - last_yaw), np.cos(target_yaw - last_yaw))
        yaw = last_yaw + yaw_diff * min(0.1, TIME_INCREMENT * 2)
        last_yaw = yaw
        
        if frame % 200 == 0:
            
            x_arr = []
            y_arr = []
            z_arr = []
            roll_arr = []
            pitch_factor = (np.sin(t / 15) + 1) / 2
            pitch = (PITCH_RANGE[0] + (PITCH_RANGE[1] - PITCH_RANGE[0]) * pitch_factor) * np.pi/180
            
            for i, (dx, dy) in enumerate(DRONE_POSITIONS):
                x_arr.append(formation_x + dx / 2.0)
                y_arr.append(formation_y + dy)
                z_arr.append(wp_z + 10 * np.sin(t / 15))
                
                curve_factor = abs(np.sin(2 * np.pi * t / 30)) * 0.5
                roll_arr.append(ROLL_RANGE[1] * curve_factor * np.sign(dx) * np.pi/180)
                
            # Pause simulation before setting poses
            client.simPause(True)
                        
            # Set poses for each drone
            for i, drone_name in enumerate(VEHICLES):
                pose = airsim.Pose(airsim.Vector3r(x_arr[i], -y_arr[i], -z_arr[i]), airsim.to_quaternion(pitch, -roll_arr[i], yaw))
                client.simSetVehiclePose(pose, True, drone_name)
            
            time.sleep(0.3)            
            client.simPause(False)
            time.sleep(0.1)
            client.simPause(True)
            time.sleep(0.3)
            
            for drone_name in VEHICLES:
                
                # Get actual drone state
                drone_state = client.getMultirotorState(vehicle_name=drone_name)
                actual_position = drone_state.kinematics_estimated.position
                actual_orientation = drone_state.kinematics_estimated.orientation
                
                # Capture images with explicit request
                responses = client.simGetImages([
                    airsim.ImageRequest(0, airsim.ImageType.Scene),
                    airsim.ImageRequest(0, airsim.ImageType.Segmentation),
                    airsim.ImageRequest(0, airsim.ImageType.DepthPlanar, True)
                ], drone_name)

                drone_path = os.path.join(base_path, drone_name)
                image_path = os.path.join(drone_path, 'images')
                
                # Use frame number in filenames to ensure uniqueness
                timestamp_str = f"{t:.3f}_{frame}"

                image_files = []
                for idx, response in enumerate(responses):
                    if idx == 0:  # scene image
                        image_files.append(os.path.join(image_path, f"img_{drone_name}_0_{t}.png"))
                        airsim.write_file(image_files[-1], response.image_data_uint8)
                    elif idx == 1:  # segmentation image
                        image_files.append(os.path.join(image_path, f"img_{drone_name}_5_{t}.png"))
                        airsim.write_file(image_files[-1], response.image_data_uint8)
                    else:  # depth image
                        image_files.append(os.path.join(image_path, f"img_{drone_name}_1_{t}.pfm"))
                        depth_image = np.array(response.image_data_float, dtype=np.float32).reshape(response.height, response.width, 1)
                        airsim.write_pfm(image_files[-1], depth_image)
            
                with open(os.path.join(drone_path, 'airsim_rec.txt'), 'a') as f:
                    image_names = ';'.join(image_files)
                    f.write(f"{drone_name}\t{t}\t{actual_position.x_val}\t{actual_position.y_val}\t{actual_position.z_val}\t{actual_orientation.w_val}\t{actual_orientation.x_val}\t{actual_orientation.y_val}\t{actual_orientation.z_val}\t{image_names}\n")

            print(f"Frame {frame}: Images captured at time {t:.2f}")
            
            # Resume simulation after all operations are complete
            client.simPause(False)
            
        t += TIME_INCREMENT
        frame += 1

finally:
    client.simPause(False)
    print("Landing...")
    tasks = []
    for vehicle in VEHICLES:
        tasks.append(client.landAsync(vehicle_name=vehicle))
    for task in tasks:
        task.join()
    for vehicle in VEHICLES:
        client.armDisarm(False, vehicle)
        client.enableApiControl(False, vehicle)
    client.reset()
