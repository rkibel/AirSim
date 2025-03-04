import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import re
import argparse
from PIL import Image

# PFM file reader function
def read_pfm(file):
    with open(file, 'rb') as f:
        # Read header
        header = f.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise ValueError('Not a PFM file.')
        
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise ValueError('Malformed PFM header.')
        
        scale = float(f.readline().decode('utf-8').rstrip())
        if scale < 0:  # little-endian
            endian = '<'
        else:
            endian = '>'
        
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        return np.reshape(data, shape)

def visualize_pfm(pfm_path, mode='jet', save_path=None, show=True, clip_percentile=None):
    """
    Visualize a PFM depth file
    
    Args:
        pfm_path: Path to the PFM file
        mode: Visualization mode ('jet', 'viridis', 'gray', 'raw')
        save_path: Path to save the visualization (optional)
        show: Whether to show the visualization
        clip_percentile: Percentile to clip values at (e.g., 95 will clip the top 5% of values)
    """
    # Read the PFM file
    depth_data = read_pfm(pfm_path)
    
    # Handle NaN or infinite values
    depth_data = np.nan_to_num(depth_data, nan=0.0, posinf=100.0, neginf=0.0)
    
    # Get depth statistics
    min_depth = np.min(depth_data)
    max_depth = np.max(depth_data)
    mean_depth = np.mean(depth_data)
    median_depth = np.median(depth_data)
    
    print(f"Depth statistics:")
    print(f"  Min: {min_depth:.4f}")
    print(f"  Max: {max_depth:.4f}")
    print(f"  Mean: {mean_depth:.4f}")
    print(f"  Median: {median_depth:.4f}")
    print(f"  Shape: {depth_data.shape}")
    
    # Clip values if requested
    if clip_percentile is not None:
        clip_value = np.percentile(depth_data, clip_percentile)
        depth_data = np.clip(depth_data, 0, clip_value)
        print(f"  Clipped at {clip_percentile}th percentile: {clip_value:.4f}")
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Visualize based on mode
    if mode == 'raw':
        # Just show the raw values
        plt.imshow(depth_data)
        plt.colorbar(label='Depth (m)')
        plt.title(f'Raw Depth Values - Min: {min_depth:.2f}m, Max: {max_depth:.2f}m')
    
    elif mode == 'gray':
        # Normalize to 0-1 and show as grayscale
        norm_depth = (depth_data - min_depth) / (max_depth - min_depth) if max_depth > min_depth else depth_data
        plt.imshow(norm_depth, cmap='gray')
        plt.colorbar(label='Normalized Depth')
        plt.title(f'Grayscale Depth - Min: {min_depth:.2f}m, Max: {max_depth:.2f}m')
    
    elif mode == 'jet' or mode == 'viridis':
        # Use a colormap for better visualization
        plt.imshow(depth_data, cmap=mode)
        plt.colorbar(label='Depth (m)')
        plt.title(f'{mode.capitalize()} Colormap - Min: {min_depth:.2f}m, Max: {max_depth:.2f}m')
    
    else:
        raise ValueError(f"Unknown visualization mode: {mode}")
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

def save_as_png(pfm_path, output_path, normalize=True, clip_percentile=None):
    """
    Convert a PFM file to PNG for easier viewing
    
    Args:
        pfm_path: Path to the PFM file
        output_path: Path to save the PNG file
        normalize: Whether to normalize the depth values to 0-255
        clip_percentile: Percentile to clip values at
    """
    # Read the PFM file
    depth_data = read_pfm(pfm_path)
    
    # Handle NaN or infinite values
    depth_data = np.nan_to_num(depth_data, nan=0.0, posinf=100.0, neginf=0.0)
    
    # Clip values if requested
    if clip_percentile is not None:
        clip_value = np.percentile(depth_data, clip_percentile)
        depth_data = np.clip(depth_data, 0, clip_value)
    
    # Normalize to 0-255 if requested
    if normalize:
        min_val = np.min(depth_data)
        max_val = np.max(depth_data)
        if max_val > min_val:
            depth_data = (depth_data - min_val) / (max_val - min_val) * 255
    else:
        # Scale to fit in 0-255 range
        depth_data = np.clip(depth_data, 0, 255)
    
    # Convert to 8-bit
    depth_img = Image.fromarray(depth_data.astype(np.uint8))
    
    # Save as PNG
    depth_img.save(output_path)
    print(f"Saved depth image to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize PFM depth files')
    parser.add_argument('pfm_path', help='Path to the PFM file')
    parser.add_argument('--mode', choices=['jet', 'viridis', 'gray', 'raw'], default='jet',
                        help='Visualization mode')
    parser.add_argument('--save', help='Path to save the visualization')
    parser.add_argument('--no-show', action='store_true', help='Do not show the visualization')
    parser.add_argument('--clip', type=float, help='Percentile to clip values at (e.g., 95)')
    parser.add_argument('--png', help='Save as PNG to this path')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.pfm_path):
        print(f"Error: File not found: {args.pfm_path}")
        return
    
    # Visualize
    visualize_pfm(
        args.pfm_path,
        mode=args.mode,
        save_path=args.save,
        show=not args.no_show,
        clip_percentile=args.clip
    )
    
    # Save as PNG if requested
    if args.png:
        save_as_png(
            args.pfm_path,
            args.png,
            normalize=True,
            clip_percentile=args.clip
        )

if __name__ == "__main__":
    main() 