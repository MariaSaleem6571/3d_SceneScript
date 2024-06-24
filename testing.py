import torch
import open3d as o3d
import pandas as pd
import numpy as np
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchsparse
import torchsparse.nn as spnn
from torchsparse import SparseTensor

# Function to read points file
def read_points_file(filepath):
    assert os.path.exists(filepath), f"Could not find point cloud file: {filepath}"
    df = pd.read_csv(filepath, compression="gzip")
    point_cloud = df[["px_world", "py_world", "pz_world"]]
    dist_std = df["dist_std"]
    print(f"Loaded point cloud with {len(point_cloud)} points.")
    return point_cloud.to_numpy(), dist_std.to_numpy()

# Example usage
points, dist_std = read_points_file("/home/mseleem/Desktop/3d_model_pt/0/semidense_points.csv.gz")

# Voxelization
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
voxel_size = 0.015  # 1.5cm
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

# Mapping Points to Voxels
voxel_map = {}
for i, point in enumerate(points):
    voxel_index = tuple(voxel_grid.get_voxel(point))
    if voxel_index in voxel_map:
        voxel_map[voxel_index].append(i)
    else:
        voxel_map[voxel_index] = [i]

# Aggregate Features for Each Voxel
all_aggregated_features = []
all_filtered_voxel_indices = []
for voxel_index, point_indices in voxel_map.items():
    aggregated_feature = np.mean(dist_std[point_indices])
    all_aggregated_features.append(aggregated_feature)
    all_filtered_voxel_indices.append(voxel_index)

# Convert to Tensors
voxel_indices_tensor = torch.tensor(all_filtered_voxel_indices, dtype=torch.int32)
features_tensor = torch.tensor(all_aggregated_features, dtype=torch.float32).view(-1, 1)
batch_indices = torch.zeros((voxel_indices_tensor.shape[0], 1), dtype=torch.int32)
voxel_indices_tensor_with_batch = torch.cat([batch_indices, voxel_indices_tensor], dim=1)

# Create custom dataset
class PointCloudDataset(Dataset):
    def __init__(self, points, features):
        self.points = points
        self.features = features

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx], self.features[idx]

# Initialize dataset and dataloader
dataset = PointCloudDataset(voxel_indices_tensor_with_batch, features_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define a simple sparse conv network for debugging
class SimpleSparseConvNet(nn.Module):
    def __init__(self):
        super(SimpleSparseConvNet, self).__init__()
        self.conv1 = spnn.Conv3d(1, 16, kernel_size=3, stride=2)

    def forward(self, x):
        try:
            print(f"Input to conv1: features shape {x.F.shape}, coordinates shape {x.C.shape}")
            x = self.conv1(x)
            print(f"Output of conv1: features shape {x.F.shape}, coordinates shape {x.C.shape}")
            return x
        except Exception as e:
            print(f"Error during forward pass: {e}")
            raise e

# Initialize the model
model = SimpleSparseConvNet()

# Forward pass using DataLoader
for batch in dataloader:
    points, features = batch
    sparse_tensor = SparseTensor(features, points)
    try:
        encoded_features = model(sparse_tensor)
        print("Forward pass successful")
    except RuntimeError as e:
        print(f"RuntimeError during encoder forward pass: {e}")
        break
    except Exception as e:
        print(f"Unexpected error: {e}")
        break
