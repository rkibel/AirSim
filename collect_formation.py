import time
import airsim
import numpy as np
import os
from datetime import datetime

# Environment boundaries (matching previous bounds)
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
SPEED = 5.0        # movement speed
PERIOD_X = 10.0    # seconds for one complete X oscillation
PERIOD_Y = 15.0    # seconds for one complete Y oscillation
DURATION = 60      # total flight duration in seconds

# Initialize AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

# Reset the environment
client.reset()
time.sleep(2)  # Wait for reset to complete

# Initialize first drone
vehicle_name = "Drone0"
client.enableApiControl(True, vehicle_name)
client.armDisarm(True, vehicle_name)

# Test basic control with first drone before proceeding
print(f"Testing control of {vehicle_name}...")
client.takeoffAsync(vehicle_name=vehicle_name).join()
client.landAsync(vehicle_name=vehicle_name).join()

print("Basic control test successful. Press Enter to continue with formation flight...")
input()

# Create and initialize rest of vehicle clients
vehicles = [vehicle_name]  # Start with the tested drone
for i in range(1, N_CAMERAS):
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

# Create timestamp and directories for recording
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
base_path = os.path.join(os.path.expanduser('~'), 'OneDrive', 'Documents', 'AirSim', timestamp)

# Create directories and start recording for each drone
for vehicle in vehicles:
    drone_path = os.path.join(base_path, vehicle)
    os.makedirs(drone_path, exist_ok=True)

# Generate formation offsets
formation_offsets = []
for row in range(FORMATION_ROWS):
    for col in range(FORMATION_COLS):
        offset_x = (col - (FORMATION_COLS-1)/2) * SPACING_X
        offset_y = (row - (FORMATION_ROWS-1)/2) * SPACING_Y
        formation_offsets.append((offset_x, offset_y))

print("Starting formation flight...")
# Main flight loop
start_time = time.time()
frame_count = 0
try:
    while time.time() - start_time < DURATION:
        t = time.time() - start_time
        frame_count += 1
        
        # Calculate formation center position
        x = wp_bound_x[0] + (wp_bound_x[1] - wp_bound_x[0]) * (0.5 + 0.5 * np.sin(2 * np.pi * t / PERIOD_X))
        y = wp_bound_y[0] + (wp_bound_y[1] - wp_bound_y[0]) * (0.5 + 0.5 * np.sin(2 * np.pi * t / PERIOD_Y))
        z = -wp_z  # Note: AirSim uses NED coordinates
        
        # Log formation data
        log_data = {
            'timestamp': t,
            'frame': frame_count,
            'formation_center_x': x,
            'formation_center_y': y,
            'formation_center_z': z
        }
        
        with open(os.path.join(base_path, 'formation_log.txt'), 'a') as f:
            if frame_count == 1:
                f.write('\t'.join(log_data.keys()) + '\n')
            f.write('\t'.join(map(str, log_data.values())) + '\n')
        
        # Move drones in formation
        tasks = []
        for vehicle, offset in zip(vehicles, formation_offsets):
            # Calculate drone position in formation
            drone_x = x + offset[0]
            drone_y = y + offset[1]
            
            # Move drone to position
            tasks.append(
                client.moveToPositionAsync(
                    drone_x, drone_y, z, 
                    SPEED,
                    vehicle_name=vehicle
                )
            )
            
            # Capture images
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene),
                airsim.ImageRequest("0", airsim.ImageType.DepthVis),
                airsim.ImageRequest("0", airsim.ImageType.Segmentation)
            ], vehicle_name=vehicle)
            
            # Save images
            image_dir = os.path.join(base_path, vehicle, 'images')
            os.makedirs(image_dir, exist_ok=True)
            for idx, response in enumerate(responses):
                filename = os.path.join(image_dir, f'frame_{frame_count:06d}_{idx}.png')
                airsim.write_file(filename, response.image_data_uint8)
        
        # Wait for all drones to reach their positions
        for task in tasks:
            task.join()
        
        time.sleep(0.05)  # 20Hz update rate

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
