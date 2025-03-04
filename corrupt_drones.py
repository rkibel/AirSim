import os
import sys
import numpy as np

sys.path.append(r"C:\Users\ronki\OneDrive\Documents\AirSim\robustness\ImageNet-C\imagenet_c")
from imagenet_c import corrupt
import PIL.Image as Image

image_path = r"C:\Users\ronki\OneDrive\Documents\AirSim\drones_images_16_GT"
corrupt_type = "shot_noise"
severities = [1, 3, 5]

for severity in severities:
    for i in range(16):
        drone = f"Drone{i}"
        print(drone)
        drone_folder = os.path.join(image_path, drone)
        drone_folder_images = os.path.join(drone_folder, f"corrupt_motion_blur_{severity}")

        corrupted_images_folder = os.path.join(drone_folder, f"corrupt_{corrupt_type}_{severity}")
        if not os.path.exists(corrupted_images_folder):
            os.makedirs(corrupted_images_folder)

        for image in os.listdir(drone_folder_images):
            if image.startswith(f"img_{drone}_0"):
                print(image)
                spec_image_path = os.path.join(drone_folder_images, image)
                img = np.array(Image.open(spec_image_path).convert('RGB'))
                corrupted_img = corrupt(img, corruption_name=corrupt_type, severity=severity)
                output_path = os.path.join(corrupted_images_folder, image)
                Image.fromarray(corrupted_img).save(output_path)
            
    