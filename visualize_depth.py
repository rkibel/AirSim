import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch_geometric.nn import GATConv
import pandas as pd
import re
import random

# Import necessary classes and functions from drone_gat_model.py
from drone_gat_model import (
    DroneDataset, 
    DepthTransform, 
    MobileNetFeatureExtractor, 
    GraphAttentionNetwork, 
    DroneReconstructionModel,
    read_pfm,
    custom_collate_fn
)

def visualize_depth_predictions(model, dataset, device, num_samples=4):
    """
    Visualize depth predictions compared to ground truth
    
    Args:
        model: Trained model
        dataset: Dataset to sample from
        device: Device to run model on
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    # Get random indices
    indices = random.sample(range(len(dataset)), num_samples)
    
    for i, idx in enumerate(indices):
        # Get sample
        rgb_img, depth_gt, position, drone_id, timestamp = dataset[idx]
        
        # Create a batch of size 1
        rgb_batch = rgb_img.unsqueeze(0).to(device)
        position_batch = position.unsqueeze(0).to(device)
        
        # Create edge index for a single node (no edges)
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        
        # Forward pass
        with torch.no_grad():
            depth_pred = model(rgb_batch, position_batch, edge_index)
        
        # Convert tensors to numpy for visualization
        rgb_np = rgb_img.permute(1, 2, 0).cpu().numpy()
        # Denormalize RGB image
        rgb_np = rgb_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        rgb_np = np.clip(rgb_np, 0, 1)
        
        # Ensure depth_gt has the right dimensions
        if depth_gt.dim() == 2:
            depth_gt = depth_gt.unsqueeze(0)
        depth_gt_np = depth_gt.squeeze().cpu().numpy()
        
        depth_pred_np = depth_pred.squeeze().cpu().numpy()
        
        # Plot
        axes[i, 0].imshow(rgb_np)
        axes[i, 0].set_title(f"RGB Image (Drone {drone_id})")
        axes[i, 0].axis('off')
        
        # Use same colormap and scale for both depth images
        vmin = min(depth_gt_np.min(), depth_pred_np.min())
        vmax = max(depth_gt_np.max(), depth_pred_np.max())
        
        im1 = axes[i, 1].imshow(depth_gt_np, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title("Ground Truth Depth")
        axes[i, 1].axis('off')
        
        im2 = axes[i, 2].imshow(depth_pred_np, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 2].set_title("Predicted Depth")
        axes[i, 2].axis('off')
        
        # Add colorbar
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        # Add timestamp and error metrics
        mse = ((depth_gt_np - depth_pred_np) ** 2).mean()
        mae = np.abs(depth_gt_np - depth_pred_np).mean()
        axes[i, 0].set_xlabel(f"Timestamp: {timestamp}")
        axes[i, 2].set_xlabel(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
    
    plt.tight_layout()
    plt.savefig("depth_predictions.png", dpi=300)
    plt.show()
    print(f"Visualization saved to depth_predictions.png")

def main():
    # Set paths and parameters
    model_path = "drone_depth_model.pth"
    data_dir = "2025-03-02-12-12-48"  # Test dataset
    input_size = 224  # MobileNet recommended size
    
    print(f"Loading model from {model_path}...")
    print(f"Using test data from {data_dir}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transforms
    transform_rgb = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats for RGB
    ])    
    transform_depth = DepthTransform(size=(input_size, input_size), normalize=True)
    
    # Initialize dataset
    print("Initializing dataset...")
    try:
        dataset = DroneDataset(data_dir, transform_rgb=transform_rgb, transform_depth=transform_depth)
        print(f"Dataset initialized successfully with {len(dataset)} samples")
        
        # Print sample counts for each drone
        for i, df in enumerate(dataset.list_dfs):
            print(f"Drone {i} has {len(df)} samples")
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return
    
    # Initialize model
    print("Initializing model...")
    try:
        model = DroneReconstructionModel(input_size=input_size, pretrained=False)  # No need for pretrained during inference
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Visualize predictions
    print("Generating visualizations...")
    try:
        visualize_depth_predictions(model, dataset, device, num_samples=4)
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main() 