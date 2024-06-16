import torch
import open3d as o3d
import pandas as pd
import numpy as np
from itertools import product
import os
from code_snippets.readers import read_points_file
from itertools import product
import torchsparse
from torch import nn
import torchsparse.nn as spnn
import torchsparse.nn.functional as F
import torchsparse.utils as sp_utils
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse import SparseTensor
from torchsparse.nn import SparseConv3d


# Read Point Clouds

def read_points_file(filepath):
    assert os.path.exists(filepath), f"Could not find point cloud file: {filepath}"
    df = pd.read_csv(filepath, compression="gzip")
    point_cloud = df[["px_world", "py_world", "pz_world"]]
    print(f"Loaded point cloud with {len(point_cloud)} points.")
    return point_cloud.to_numpy()

points= read_points_file("/home/mseleem/Desktop/3d_model_pt/0/semidense_points.csv.gz")

# Voxelize
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
voxel_size = 0.05  # 5cm
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
o3d.visualization.draw_geometries([voxel_grid])

voxel_centers = []
for voxel in voxel_grid.get_voxels():
    center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
    voxel_centers.append(center)

voxel_centers = np.array(voxel_centers)

print(voxel_centers.shape)

def get_voxel_vertices(center, edge_length):
    half_edge_length = edge_length / 2
    offsets = np.array(list(product([-half_edge_length, half_edge_length], repeat=3)))
    vertices = center + offsets
    return vertices 

voxel_vertices = [get_voxel_vertices(center, voxel_size) for center in voxel_centers]

# Feature channels (3 for coordinates)
feature_channels = 3

sparse_tensor = SparseTensor(
    coords=torch.from_numpy(voxel_centers).float(),
    features=torch.zeros(len(voxel_centers), feature_channels).float(),
) 

# Series of SparseConv3d layers for down convolutions
class SparseConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SparseConvEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            spnn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            spnn.BatchNorm(32),
            spnn.ReLU(True),
            spnn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            spnn.BatchNorm(64),
            spnn.ReLU(True),
            spnn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            spnn.BatchNorm(128),
            spnn.ReLU(True),
            spnn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            spnn.BatchNorm(256),
            spnn.ReLU(True),
            spnn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1),
            spnn.BatchNorm(512),
            spnn.ReLU(True),
        )
        self.output_layer = spnn.Conv3d(512, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.output_layer(x)
        return x

encoder = SparseConvEncoder(in_channels=feature_channels, out_channels=512)

# Get pooled features
pooled_features = encoder(sparse_tensor)
print(f"F_geo (pooled_features) shape: {pooled_features.shape}")

sorted_features, sorted_indices = torch.sparse.sort_along_dim(
    pooled_features, dim=0, index=voxel_centers
)

dense_features = sorted_features.to_dense()

#dense_features -> dense tensor of shape (K, feature_channels)
#sorted_indices -> dense tensor of voxel center coordinates (K, 3)

# Concatenate features with voxel coordinates
encoded_features = torch.cat([dense_features, sorted_indices], dim=1)


# class SparseConvEncoder(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(SparseConvEncoder, self).__init__()
#         self.conv_layers = nn.Sequential(
#             spnn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
#             spnn.BatchNorm(32),
#             spnn.ReLU(True),
#             spnn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
#             spnn.BatchNorm(64),
#             spnn.ReLU(True),
#             spnn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
#             spnn.BatchNorm(128),
#             spnn.ReLU(True),
#             spnn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
#             spnn.BatchNorm(256),
#             spnn.ReLU(True),
#             spnn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1),
#             spnn.BatchNorm(512),
#             spnn.ReLU(True),
#         )
#         self.output_layer = spnn.Conv3d(512, out_channels, kernel_size=1, stride=1)

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = self.output_layer(x)
#         return x

# # Initialize model
# model = SparseConvEncoder(in_channels=3, out_channels=512)

# # Encode the input voxel data
# encoded_features = model(input_tensor)

# # To get features and coordinates from SparseTensor
# encoded_features_coords = encoded_features.C
# encoded_features_features = encoded_features.F

# # Sorting and combining with coordinates if necessary for positional encoding
# sorted_indices = torch.argsort(encoded_features_coords[:, 0], dim=0)
# sorted_features = encoded_features_features[sorted_indices]
# sorted_coordinates = encoded_features_coords[sorted_indices]

# # Concatenating features with coordinates
# final_features = torch.cat([sorted_features, sorted_coordinates.float()], dim=1)

# print("Encoded and positionally encoded features:", final_features)

