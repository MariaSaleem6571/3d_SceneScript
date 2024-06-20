# %% [markdown]
# **Point Cloud Encoder**

# %% [markdown]
# *Importing Libraries*

# %%
import torch
import open3d as o3d
import pandas as pd
import numpy as np
import os
from code_snippets.readers import read_points_file
import torchsparse
from torch import nn
import torchsparse.nn as spnn
import torchsparse.nn.functional as F
import torchsparse.utils as sp_utils
from torchsparse import SparseTensor


# %% [markdown]
# *Initializing the Point Clouds Data*

# %%
def read_points_file(filepath):
    assert os.path.exists(filepath), f"Could not find point cloud file: {filepath}"
    df = pd.read_csv(filepath, compression="gzip")
    point_cloud = df[["px_world", "py_world", "pz_world"]]
    dist_std = df["dist_std"]
    print(f"Loaded point cloud with {len(point_cloud)} points.")
    return point_cloud.to_numpy(), dist_std.to_numpy()

# Example usage
points, dist_std = read_points_file("/home/mseleem/Desktop/3d_model_pt/0/semidense_points.csv.gz")

# %% [markdown]
# *Voxelize Point Clouds*

# %%
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# %%
num_points = len(pcd.points)
print(f"Number of points: {num_points}")

# %%
voxel_size = 0.015  # 1.5cm
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

# %%
print("Number of voxels:", len(voxel_grid.get_voxels()))

# %%
o3d.visualization.draw_geometries([voxel_grid])

# %%
# Initialize a dictionary to map voxel indices to point indices
voxel_map = {}

# Map points to their respective voxels
for i, point in enumerate(points):
    voxel_index = tuple(voxel_grid.get_voxel(point))
    if voxel_index in voxel_map:
        voxel_map[voxel_index].append(i)
    else:
        voxel_map[voxel_index] = [i]

# %%
# Check for empty voxels
empty_voxels = [k for k, v in voxel_map.items() if len(v) == 0]
assert len(empty_voxels) == 0, "There are empty voxels in the mapping!"
print("All voxels have associated points.")

# %%
# Check total points
total_points_in_voxels = sum(len(v) for v in voxel_map.values())
print(f"Total points in voxels: {total_points_in_voxels}")
print(f"Total original points: {len(points)}")
assert total_points_in_voxels == len(points), f"Mismatch in point counts: {total_points_in_voxels} != {len(points)}"

# %%
# Aggregate features for each voxel
aggregated_features = []
filtered_voxel_indices = []
for voxel_index, point_indices in voxel_map.items():
    aggregated_feature = np.mean(dist_std[point_indices])  # Example aggregation: mean
    aggregated_features.append(aggregated_feature)
    filtered_voxel_indices.append(voxel_index)

# %%
# Print a few voxel mappings for verification
print("Sample voxel mappings (first 10):")
for k, v in list(voxel_map.items())[:10]:
    print(f"Voxel index: {k}, Point indices: {v}")

# %%
# Verify feature aggregation for a few voxels
print("\nAggregated features for sample voxels (first 10):")
aggregated_features = []
filtered_voxel_indices = []
for voxel_index, point_indices in list(voxel_map.items())[:10]:  # Check first 10 voxels
    aggregated_feature = np.mean(dist_std[point_indices])  # Example aggregation: mean
    print(f"Voxel index: {voxel_index}, Aggregated feature: {aggregated_feature:.4f}")
    aggregated_features.append(aggregated_feature)
    filtered_voxel_indices.append(voxel_index)

# %%
# Convert to tensor (for the full dataset)
all_aggregated_features = []
all_filtered_voxel_indices = []
for voxel_index, point_indices in voxel_map.items():
    aggregated_feature = np.mean(dist_std[point_indices])
    all_aggregated_features.append(aggregated_feature)
    all_filtered_voxel_indices.append(voxel_index)

# %%
# Convert to tensor
voxel_indices_tensor = torch.tensor(all_filtered_voxel_indices, dtype=torch.int32)
features_tensor = torch.tensor(all_aggregated_features, dtype=torch.float32).view(-1, 1)

# %%
# Add batch dimension to voxel indices
batch_indices = torch.zeros((voxel_indices_tensor.shape[0], 1), dtype=torch.int32)
voxel_indices_tensor_with_batch = torch.cat([batch_indices, voxel_indices_tensor], dim=1)

# %%
# Print dimensions of the tensors
print(f"Voxel indices tensor dimensions: {voxel_indices_tensor_with_batch.shape}")
print(f"Features tensor dimensions: {features_tensor.shape}")

# %%
# Create sparse tensor
sparse_tensor = SparseTensor(features_tensor, voxel_indices_tensor_with_batch)

# Print dimensions of the sparse tensor
print(f"Sparse tensor feature dimensions: {sparse_tensor.F.shape}")
print(f"Sparse tensor coordinate dimensions: {sparse_tensor.C.shape}")

# %%
class SparseResNetEncoder(nn.Module):
    def __init__(self):
        super(SparseResNetEncoder, self).__init__()
        self.conv1 = spnn.Conv3d(1, 16, kernel_size=3, stride=2)
        self.conv2 = spnn.Conv3d(16, 32, kernel_size=3, stride=2)
        self.conv3 = spnn.Conv3d(32, 64, kernel_size=3, stride=2)
        self.conv4 = spnn.Conv3d(64, 128, kernel_size=3, stride=2)
        self.conv5 = spnn.Conv3d(128, 512, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


# %%
# Initialize the model
encoder = SparseResNetEncoder()

# Forward pass
encoded_features = encoder(sparse_tensor)

# Append coordinates to feature vectors for positional encoding
# coordinates = sparse_tensor.C.float()
# encoded_features_with_pos = torch.cat([encoded_features.F, coordinates], dim=1)

# Sort the feature vectors lexicographically by coordinates
# sorted_indices = torch.lexsort((coordinates[:, 2], coordinates[:, 1], coordinates[:, 0]))
# sorted_features_with_pos = encoded_features_with_pos[sorted_indices]

# print("Encoded and sorted features with positional encoding:")
# print(sorted_features_with_pos)


