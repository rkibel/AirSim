import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch_geometric.nn import GATConv
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import re
import glob
import torch.utils.data.sampler as sampler
import random

# Add this PFM file reader function at the top of your file
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

# Add a custom batch sampler that groups drones by timestamp
class TimestampBatchSampler(sampler.Sampler):
    def __init__(self, dataset, batch_size=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.timestamps = sorted(list(dataset.timestamp_to_index.keys()))
        
        # Create mapping from timestamp to indices for each drone
        self.timestamp_to_drone_indices = {}
        for ts in self.timestamps:
            self.timestamp_to_drone_indices[ts] = []
            
            # Find indices for each drone at this timestamp
            for drone_id, df in enumerate(dataset.list_dfs):
                # Find rows in this drone's dataframe with this timestamp
                matching_rows = df[df['TimeStamp'] == ts]
                if not matching_rows.empty:
                    # Get the index within this drone's dataframe
                    row_idx = matching_rows.index[0]
                    # Calculate the global index
                    global_idx = dataset.get_global_index(drone_id, row_idx)
                    self.timestamp_to_drone_indices[ts].append(global_idx)
        
        # Filter out timestamps that don't have enough drones
        self.valid_timestamps = [ts for ts in self.timestamps 
                               if len(self.timestamp_to_drone_indices[ts]) >= self.batch_size]
        
        print(f"Found {len(self.valid_timestamps)} valid timestamps with at least {self.batch_size} drones")
    
    def __iter__(self):
        # Shuffle the timestamps
        random_timestamps = self.valid_timestamps.copy()
        random.shuffle(random_timestamps)
        
        # For each timestamp, yield the indices of all drones at that timestamp
        for ts in random_timestamps:
            yield self.timestamp_to_drone_indices[ts]
    
    def __len__(self):
        return len(self.valid_timestamps)

# Define your dataset class for the drone data
class DroneDataset(Dataset):
    def __init__(self, data_dir, transform_rgb=None, transform_gray=None):
        self.data_dir = data_dir
        self.transform_rgb = transform_rgb
        self.transform_gray = transform_gray
        
        print(f"Looking for drone folders in: {data_dir}")
        
        # each entry in list_dfs now contains TimeStamp, POS_X, POS_Y, POS_Z, Q_W, Q_X, Q_Y, Q_Z, rgb, seg, depth
        self.list_dfs = []
        
        # Find all drone folders
        drone_folders = [d for d in os.listdir(data_dir) if d.startswith('Drone') and os.path.isdir(os.path.join(data_dir, d))]
        print(f"Found {len(drone_folders)} drone folders: {drone_folders}")
        
        # Process each drone folder
        for drone_folder in drone_folders:
            rec_file = os.path.join(data_dir, drone_folder, 'airsim_rec.txt')
            df = pd.read_csv(rec_file, sep='\t')
            df[['rgb', 'seg', 'depth']] = df['ImageFile'].str.split(';', expand=True)
            self.list_dfs.append(df)
        
        # Create timestamp to index mapping from the first drone's dataframe
        # This will be used to synchronize drones by timestamp
        self.timestamp_to_index = {row['TimeStamp']: idx for idx, row in self.list_dfs[0].iterrows()}
        
        # Calculate cumulative lengths for efficient indexing
        self.cumulative_lengths = [0]
        for df in self.list_dfs:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(df))
    
    def get_global_index(self, drone_id, row_idx):
        """Convert a (drone_id, row_idx) pair to a global index"""
        return self.cumulative_lengths[drone_id] + row_idx
    
    def get_drone_and_row(self, idx):
        """Convert a global index to a (drone_id, row_idx) pair"""
        drone_id = 0
        remaining_idx = idx
        
        for i, df in enumerate(self.list_dfs):
            if remaining_idx < len(df):
                drone_id = i
                break
            remaining_idx -= len(df)
        
        return drone_id, remaining_idx

    def __len__(self):
        # Return the total number of samples across all drone dataframes
        return sum(len(df) for df in self.list_dfs)

    def __getitem__(self, idx):
        # Find which drone dataframe contains this index
        drone_id, remaining_idx = self.get_drone_and_row(idx)
        
        # Get the row from the appropriate dataframe
        row = self.list_dfs[drone_id].iloc[remaining_idx]
        
        # Extract position data
        position = [row['POS_X'], row['POS_Y'], row['POS_Z']]
        position_tensor = torch.tensor(position, dtype=torch.float)
        
        # Get image paths
        rgb_path = os.path.join(self.data_dir, row['rgb'])
        depth_path = os.path.join(self.data_dir, row['depth'])
        seg_path = os.path.join(self.data_dir, row['seg'])
        
        # Load RGB image
        rgb_img = Image.open(rgb_path).convert('RGB')
        
        # Load depth as PFM
        if os.path.exists(depth_path) and depth_path.endswith('.pfm'):
            depth_data = read_pfm(depth_path)
            # Normalize depth data to 0-1 range for easier processing
            depth_min, depth_max = np.min(depth_data), np.max(depth_data)
            if depth_max > depth_min:
                depth_data = (depth_data - depth_min) / (depth_max - depth_min)
            depth_img = Image.fromarray((depth_data * 255).astype(np.uint8))
        else:
            # Try to load as regular image if not PFM
            try:
                depth_img = Image.open(depth_path).convert('L')
            except:
                print(f"Warning: Depth file not found or cannot be read: {depth_path}")
                depth_img = Image.new('L', rgb_img.size, 0)  # Create blank depth image
        
        # Load segmentation
        try:
            seg_img = Image.open(seg_path).convert('L')
        except:
            print(f"Error loading segmentation: {seg_path}")
            seg_img = Image.new('L', rgb_img.size, 0)  # Create blank segmentation
        
        # Apply transforms
        if self.transform_rgb:
            rgb_img = self.transform_rgb(rgb_img)
        
        if self.transform_gray:
            depth_img = self.transform_gray(depth_img)
            seg_img = self.transform_gray(seg_img)
        
        # Also return the timestamp for synchronization
        timestamp = row['TimeStamp']
        
        return rgb_img, depth_img, seg_img, position_tensor, drone_id, timestamp

# Replace CNNFeatureExtractor with MobileNetFeatureExtractor
class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetFeatureExtractor, self).__init__()
        # Load MobileNetV2 with or without pretrained weights
        if pretrained:
            weights = MobileNet_V2_Weights.DEFAULT
        else:
            weights = None
            
        mobilenet = mobilenet_v2(weights=weights)
        
        # Use MobileNet features except the classifier
        self.features = nn.Sequential(*list(mobilenet.features))
    
    def forward(self, x):
        return self.features(x)

# Add this missing GraphAttentionNetwork class definition
class GraphAttentionNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super(GraphAttentionNetwork, self).__init__()
        self.gat = GATConv(in_channels, out_channels // heads, heads=heads, concat=True)
    
    def forward(self, x, edge_index):
        return self.gat(x, edge_index)

# Modify the DroneReconstructionModel to use MobileNet
class DroneReconstructionModel(nn.Module):
    def __init__(self, input_size=224, pretrained=False):  # MobileNet works better with 224x224 input
        super(DroneReconstructionModel, self).__init__()
        self.cnn = MobileNetFeatureExtractor(pretrained=pretrained)
        
        # MobileNetV2 outputs 1280 channels with spatial dimensions input_size/32
        spatial_size = input_size // 32
        feature_size = 1280 * spatial_size * spatial_size
        
        # MLP to process position data
        self.position_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # GAT for feature processing
        self.gat = GraphAttentionNetwork(in_channels=feature_size + 128, out_channels=1024)
        
        # Decoder for depth prediction (single channel output)
        self.depth_decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, input_size * input_size),
            nn.Sigmoid()  # Normalize depth values
        )
        
        self.input_size = input_size

    def forward(self, x, positions, edge_index, target_indices=None):
        """
        Forward pass through the model
        
        Args:
            x: Tensor of RGB images [batch_size, 3, H, W]
            positions: Tensor of drone positions [batch_size, 3]
            edge_index: Tensor defining graph connections [2, num_edges]
            target_indices: Indices of target drones to make predictions for (default: all drones)
            
        Returns:
            depth_output: Predicted depth maps for target drones
        """
        batch_size = x.size(0)
        
        # Extract image features
        image_features = self.cnn(x)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten
        
        # Process position data
        pos_features = self.position_encoder(positions)
        
        # Concatenate image and position features
        combined_features = torch.cat([image_features, pos_features], dim=1)
        
        # Apply GAT to get features for all drones
        all_gat_features = self.gat(combined_features, edge_index)
        
        # If target_indices is None, make predictions for all drones
        if target_indices is None:
            target_indices = list(range(batch_size))
        
        # Extract features for target drones only
        target_features = all_gat_features[target_indices]
        
        # Decode to depth for target drones only
        depth_output = self.depth_decoder(target_features)
        depth_output = depth_output.view(-1, 1, self.input_size, self.input_size)  # Add channel dimension
        
        return depth_output

# Training loop
def train_model(model, dataloader, criterion_depth, optimizer, device, num_epochs=50, patience=10, validation_split=0.2):
    """
    Train the model with early stopping
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        criterion_depth: Loss function for depth prediction
        optimizer: Optimizer for training
        device: Device to train on (cuda/cpu)
        num_epochs: Maximum number of epochs to train for
        patience: Number of epochs to wait for improvement before stopping
        validation_split: Fraction of data to use for validation
    """
    model.to(device)
    
    # Split data into training and validation
    dataset_size = len(dataloader.batch_sampler)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    
    # Shuffle indices
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    print(f"Training on {len(train_indices)} batches, validating on {len(val_indices)} batches")
    
    # Variables for early stopping
    best_val_loss = float('inf')
    no_improve_epochs = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        total_train_samples = 0
        
        # Track which batches to use for training
        batch_idx = 0
        
        for batch_data in dataloader:
            # Skip this batch if it's in the validation set
            if batch_idx in val_indices:
                batch_idx += 1
                continue
                
            # Unpack batch data
            rgb_imgs, depth_imgs, _, positions, drone_ids, timestamps = batch_data
            
            # Move data to device
            rgb_imgs = rgb_imgs.to(device)
            depth_imgs = depth_imgs.to(device)
            positions = positions.to(device)
            drone_ids = drone_ids.to(device)
            
            batch_size = rgb_imgs.size(0)
            total_train_samples += batch_size
            
            # Create edge index for a fully connected graph
            edge_index = torch.zeros((2, batch_size * (batch_size - 1)), dtype=torch.long, device=device)
            idx = 0
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:  # Don't connect a drone to itself
                        edge_index[0, idx] = i  # Source node
                        edge_index[1, idx] = j  # Target node
                        idx += 1
            
            # Forward pass
            optimizer.zero_grad()
            depth_pred = model(rgb_imgs, positions, edge_index)
            
            # Calculate loss
            depth_loss = criterion_depth(depth_pred, depth_imgs)
            batch_loss = depth_loss
            
            # Backward pass
            batch_loss.backward()
            optimizer.step()
            
            running_train_loss += batch_loss.item() * batch_size
            
            # Print batch statistics
            if batch_size > 0:
                print(f"Epoch {epoch+1}, Train Batch {batch_idx}, Loss: {batch_loss.item():.4f}, "
                      f"Timestamp: {timestamps[0]}, Drones: {drone_ids.tolist()}")
            
            batch_idx += 1
        
        # Calculate training epoch loss
        train_epoch_loss = running_train_loss / total_train_samples if total_train_samples > 0 else 0
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        total_val_samples = 0
        batch_idx = 0
        
        with torch.no_grad():  # No gradients needed for validation
            for batch_data in dataloader:
                # Only use batches in the validation set
                if batch_idx not in val_indices:
                    batch_idx += 1
                    continue
                    
                # Unpack batch data
                rgb_imgs, depth_imgs, _, positions, drone_ids, timestamps = batch_data
                
                # Move data to device
                rgb_imgs = rgb_imgs.to(device)
                depth_imgs = depth_imgs.to(device)
                positions = positions.to(device)
                
                batch_size = rgb_imgs.size(0)
                total_val_samples += batch_size
                
                # Create edge index for a fully connected graph
                edge_index = torch.zeros((2, batch_size * (batch_size - 1)), dtype=torch.long, device=device)
                idx = 0
                for i in range(batch_size):
                    for j in range(batch_size):
                        if i != j:  # Don't connect a drone to itself
                            edge_index[0, idx] = i  # Source node
                            edge_index[1, idx] = j  # Target node
                            idx += 1
                
                # Forward pass
                depth_pred = model(rgb_imgs, positions, edge_index)
                
                # Calculate loss
                depth_loss = criterion_depth(depth_pred, depth_imgs)
                batch_loss = depth_loss
                
                running_val_loss += batch_loss.item() * batch_size
                
                # Print batch statistics
                if batch_size > 0:
                    print(f"Epoch {epoch+1}, Val Batch {batch_idx}, Loss: {batch_loss.item():.4f}, "
                          f"Timestamp: {timestamps[0]}, Drones: {drone_ids.tolist()}")
                
                batch_idx += 1
        
        # Calculate validation epoch loss
        val_epoch_loss = running_val_loss / total_val_samples if total_val_samples > 0 else float('inf')
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}')
        
        # Check if validation loss improved
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            no_improve_epochs = 0
            best_model_state = model.state_dict().copy()
            print(f"Validation loss improved to {best_val_loss:.4f}")
        else:
            no_improve_epochs += 1
            print(f"Validation loss did not improve. No improvement for {no_improve_epochs} epochs.")
            
        # Early stopping check
        if no_improve_epochs >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            # Restore best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break
    
    # Restore best model if we completed all epochs
    if best_model_state is not None and no_improve_epochs > 0:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with validation loss: {best_val_loss:.4f}")
    
    return model, best_val_loss

# Update the main function to use early stopping
def main():
    data_dir = "C:/Users/ronki/OneDrive/Documents/AirSim/2025-03-02-12-12-48"
    input_size = 224  # MobileNet recommended size
    batch_size = 4  # Number of drones to process together
    use_pretrained = False  # Set to False to avoid downloading pretrained weights
    num_epochs = 50  # Maximum number of epochs
    patience = 10  # Number of epochs to wait before early stopping
    
    print("\n" + "="*50)
    print("Starting drone reconstruction model training (DEPTH ONLY)")
    print("="*50 + "\n")
    
    print(f"Data directory: {data_dir}")
    print(f"Input size: {input_size}x{input_size}")
    print(f"Batch size: {batch_size}")
    print(f"Using pretrained weights: {use_pretrained}")
    print(f"Maximum epochs: {num_epochs}")
    print(f"Early stopping patience: {patience}")
    
    # Create separate transforms for RGB and grayscale images
    print("\nCreating image transforms...")
    transform_rgb = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats for RGB
    ])
    
    transform_gray = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        # No normalization for grayscale images, or use single-channel normalization if needed
        # transforms.Normalize(mean=[0.5], std=[0.5])  # Optional simple normalization for grayscale
    ])
    print("Transforms created successfully.")
    
    # Initialize dataset with the existing DroneDataset class
    print("\nInitializing dataset...")
    try:
        dataset = DroneDataset(data_dir, transform_rgb=transform_rgb, transform_gray=transform_gray)
        print("Dataset initialized successfully.")
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return
    
    # Print dataset information for debugging
    print(f"\nDataset initialized with {len(dataset)} samples")
    print(f"Number of drone dataframes: {len(dataset.list_dfs)}")
    
    # Print sample counts for each drone
    for i, df in enumerate(dataset.list_dfs):
        print(f"Drone {i} has {len(df)} samples")
    
    # Verify timestamp_to_index dictionary
    if dataset.timestamp_to_index:
        print(f"timestamp_to_index contains {len(dataset.timestamp_to_index)} entries")
        # Print a few sample entries
        sample_timestamps = list(dataset.timestamp_to_index.keys())[:3]
        for ts in sample_timestamps:
            print(f"Timestamp {ts} maps to index {dataset.timestamp_to_index[ts]}")
    else:
        print("Warning: timestamp_to_index is None or empty")
    
    # Test getting a few items from the dataset
    print("\nTesting dataset item retrieval...")
    if len(dataset) > 0:
        try:
            print("Testing __getitem__ with index 0:")
            rgb, depth, seg, pos, drone_id, timestamp = dataset[0]
            print(f"  RGB shape: {rgb.shape}, Depth shape: {depth.shape}, Seg shape: {seg.shape}")
            print(f"  Position: {pos}, Drone ID: {drone_id}, Timestamp: {timestamp}")
        except Exception as e:
            print(f"Error retrieving item 0: {e}")
    
    if len(dataset) > 1:
        try:
            print("Testing __getitem__ with index 1:")
            rgb, depth, seg, pos, drone_id, timestamp = dataset[1]
            print(f"  RGB shape: {rgb.shape}, Depth shape: {depth.shape}, Seg shape: {seg.shape}")
            print(f"  Position: {pos}, Drone ID: {drone_id}, Timestamp: {timestamp}")
        except Exception as e:
            print(f"Error retrieving item 1: {e}")
    
    # Create timestamp batch sampler
    print("\nCreating timestamp batch sampler...")
    try:
        batch_sampler = TimestampBatchSampler(dataset, batch_size=batch_size)
        print(f"Batch sampler created with {len(batch_sampler)} valid timestamps")
    except Exception as e:
        print(f"Error creating batch sampler: {e}")
        return
    
    # Create dataloader with the custom batch sampler
    print("\nCreating dataloader with timestamp batch sampler...")
    try:
        dataloader = DataLoader(
            dataset, 
            batch_sampler=batch_sampler,
            num_workers=0  # Set to 0 for debugging
        )
        print("Dataloader created successfully.")
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Initialize model
    print("\nInitializing model...")
    try:
        model = DroneReconstructionModel(input_size=input_size, pretrained=use_pretrained)
        print("Model initialized successfully.")
        
        # Print model summary
        print("\nModel architecture:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    # Define loss function and optimizer
    print("\nSetting up loss function and optimizer...")
    criterion_depth = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Loss function and optimizer set up successfully.")
    
    # Train the model with early stopping
    print("\nStarting training (depth only) with early stopping...")
    try:
        model, best_val_loss = train_model(
            model, 
            dataloader, 
            criterion_depth, 
            optimizer, 
            device, 
            num_epochs=num_epochs,
            patience=patience
        )
        print(f"Training completed successfully. Best validation loss: {best_val_loss:.4f}")
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    # Save the model
    print("\nSaving model...")
    try:
        model_save_path = "C:/Users/ronki/OneDrive/Documents/AirSim/drone_depth_model.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    main()
