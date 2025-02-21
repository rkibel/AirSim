import time
import airsim
import numpy as np

# Set boundary parameters
wp_bound_x = [-100, 100]
wp_bound_y = [-100, 100]
wp_z = 50

# Flight parameters
SPEED = 5.0        # movement speed in m/s
DURATION = 35     # total flight duration in seconds
SAMPLE_TIME = 0.1  # time between pose updates

# Camera parameters
PITCH_RANGE = [-90, -45]  # degrees, -90 is down, -45 is forward-down
ROLL_RANGE = [-20, 20]    # degrees, for banking turns
YAW_RATE = 1.0           # reduced for smoother rotation

# Set initial pose
x = wp_bound_x[0]
y = wp_bound_y[0]
z = wp_z
pitch = PITCH_RANGE[0] * np.pi/180
roll = 0
yaw = 0
last_yaw = 0  # Track previous yaw for smoothing

# Connect to AirSim
client = airsim.VehicleClient()
client.reset()
client.confirmConnection()

# Start recording
client.startRecording()

# === Generate Smooth Trajectory ===
start_time = time.time()
t = 0

try:
    while t < DURATION:
        # Calculate current time
        t = time.time() - start_time
        
        # Generate smooth figure-8 pattern
        scale_x = (wp_bound_x[1] - wp_bound_x[0]) / 2
        scale_y = (wp_bound_y[1] - wp_bound_y[0]) / 2
        center_x = (wp_bound_x[1] + wp_bound_x[0]) / 2
        center_y = (wp_bound_y[1] + wp_bound_y[0]) / 2
        
        # Position equations for a Lissajous curve (modified figure-8)
        x = center_x + scale_x * np.sin(2 * np.pi * t / 30)
        y = center_y + scale_y * np.sin(4 * np.pi * t / 30)
        
        # Calculate yaw with continuous rotation handling
        dx = scale_x * (2 * np.pi / 30) * np.cos(2 * np.pi * t / 30)
        dy = scale_y * (4 * np.pi / 30) * np.cos(4 * np.pi * t / 30)
        target_yaw = np.arctan2(dy, dx)
        
        # Ensure smooth yaw transitions across the -pi/pi boundary
        yaw_diff = np.arctan2(np.sin(target_yaw - last_yaw), np.cos(target_yaw - last_yaw))
        yaw = last_yaw + yaw_diff * min(0.1, SAMPLE_TIME * 2)
        last_yaw = yaw
        
        # Smoother pitch variations
        pitch_factor = (np.sin(t / 15) + 1) / 2
        pitch = (PITCH_RANGE[0] + (PITCH_RANGE[1] - PITCH_RANGE[0]) * pitch_factor) * np.pi/180
        
        # Smoother banking turns
        curve_factor = abs(np.sin(2 * np.pi * t / 30))
        roll = ROLL_RANGE[1] * curve_factor * np.sign(dx) * np.pi/180
        
        # Add slight altitude variation
        z = wp_z + 10 * np.sin(t / 15)  # Oscillate ±10m from base altitude
        
        # Set the pose
        pose = airsim.Pose(
            airsim.Vector3r(x, -y, -z),
            airsim.to_quaternion(pitch, -roll, yaw)
        )
        client.simSetVehiclePose(pose, True)
        
        time.sleep(SAMPLE_TIME)

finally:
    # Stop recording
    client.stopRecording() 