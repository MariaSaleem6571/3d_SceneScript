import torch
import open3d as o3d
import pandas as pd
import numpy as np
import os
from torch import nn
import torchsparse
import torchsparse.nn as spnn
from torchsparse import SparseTensor

class PointCloudTransformerLayer(nn.Module):
    """
    Point Cloud Transformer Layer

    Args:
        voxel_size (float): The voxel size for the point cloud
        emb_dim (int): The embedding dimension

    Returns:
        encoded_features_with_pos
    """

    def __init__(self, voxel_size=0.03, emb_dim=512, max_positions=7000):
        super(PointCloudTransformerLayer, self).__init__()
        self.voxel_size = voxel_size
        self.emb_dim = emb_dim
        self.sparse_encoder = SparseResNetEncoder()

        self.positional_embeddings = nn.Embedding(max_positions, emb_dim).to('cuda')

    def set_voxel_size(self, new_voxel_size):
        """Set a new voxel size dynamically."""
        self.voxel_size = new_voxel_size

    @classmethod
    def read_points_file(cls, filepath):
        """
        Read the point cloud file

        Args:
            filepath (str): The path to the point cloud file

        Returns:
            point_cloud, dist_std
        """

        assert os.path.exists(filepath), f"Could not find point cloud file: {filepath}"
        df = pd.read_csv(filepath, compression="gzip")
        point_cloud = df[["px_world", "py_world", "pz_world"]]
        dist_std = df["dist_std"]
        return point_cloud.to_numpy(), dist_std.to_numpy()


    def process_point_cloud(self, points, dist_std):
        """
        Process the point cloud

        Args:
            points (np.ndarray): The points
            dist_std (np.ndarray): The distance standard deviation

        Returns:
            sparse_tensor
        """

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, self.voxel_size)

        voxel_map = {}
        for i, point in enumerate(points):
            voxel_index = tuple(voxel_grid.get_voxel(point))
            if voxel_index in voxel_map:
                voxel_map[voxel_index].append(i)
            else:
                voxel_map[voxel_index] = [i]

        aggregated_features = []
        filtered_voxel_indices = []
        for idx, (voxel_index, point_indices) in enumerate(voxel_map.items()):
            aggregated_feature = np.mean(dist_std[point_indices])
            aggregated_features.append(aggregated_feature)
            filtered_voxel_indices.append(voxel_index)

        voxel_indices_tensor = torch.tensor(filtered_voxel_indices, dtype=torch.int32, device='cuda')
        features_tensor = torch.tensor(aggregated_features, dtype=torch.float32, device='cuda').view(-1, 1)

        batch_indices = torch.zeros((voxel_indices_tensor.shape[0], 1), dtype=torch.int32, device='cuda')
        voxel_indices_tensor_with_batch = torch.cat([batch_indices, voxel_indices_tensor], dim=1)

        sparse_tensor = SparseTensor(features_tensor, voxel_indices_tensor_with_batch)
        return sparse_tensor

    def forward(self, sparse_tensor):
        """
        Forward pass

        Args:
            sparse_tensor: The sparse tensor

        Returns:
            encoded_features_with_pos
        """

        encoded_features = self.sparse_encoder(sparse_tensor)
        positions = torch.arange(0, encoded_features.C.shape[0], device='cuda')
        positional_encodings = self.positional_embeddings(positions)
        encoded_features_with_pos = encoded_features.F + positional_encodings

        return encoded_features_with_pos

class SparseResNetEncoder(nn.Module):
    def __init__(self):
        super(SparseResNetEncoder, self).__init__()
        self.conv1 = spnn.Conv3d(1, 16, kernel_size=3, stride=2)
        self.conv2 = spnn.Conv3d(16, 32, kernel_size=3, stride=2)
        self.conv3 = spnn.Conv3d(32, 64, kernel_size=3, stride=2)
        self.conv4 = spnn.Conv3d(64, 128, kernel_size=3, stride=2)
        self.conv5 = spnn.Conv3d(128, 512, kernel_size=3, stride=2)

    def forward(self, x):
        x = x.to('cuda')
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x
