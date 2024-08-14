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
    def __init__(self, voxel_size=0.03, emb_dim=512):
        super(PointCloudTransformerLayer, self).__init__()
        self.voxel_size = voxel_size
        self.emb_dim = emb_dim
        
        self.sparse_encoder = SparseResNetEncoder()

    @classmethod
    def read_points_file(cls, filepath):
        assert os.path.exists(filepath), f"Could not find point cloud file: {filepath}"
        df = pd.read_csv(filepath, compression="gzip")
        point_cloud = df[["px_world", "py_world", "pz_world"]]
        dist_std = df["dist_std"]
        return point_cloud.to_numpy(), dist_std.to_numpy()

    def generate_sinusoidal_positional_encoding(self, coordinates, d_model):
        n_positions, n_dims = coordinates.shape
        pe = torch.zeros(n_positions, d_model, device='cuda')
        position = coordinates.float().cuda()
        div_term = torch.exp(torch.arange(0, d_model // n_dims, 2).float() * -(np.log(10000.0) / (d_model // n_dims))).cuda()

        for i in range(n_dims):
            pe[:, 2 * i:d_model:2 * n_dims] = torch.sin(position[:, i].unsqueeze(1) * div_term)
            pe[:, 2 * i + 1:d_model:2 * n_dims] = torch.cos(position[:, i].unsqueeze(1) * div_term)

        return pe

    def process_point_cloud(self, points, dist_std):
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
        encoded_features = self.sparse_encoder(sparse_tensor)
        
        positional_encoding = self.generate_sinusoidal_positional_encoding(encoded_features.C, self.emb_dim)
        
        encoded_features_with_pos = encoded_features.F + positional_encoding
        
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

if __name__ == '__main__':
    model = PointCloudTransformerLayer().cuda()
    pt_cloud_path = "/home/mseleem/Desktop/3d_model_pt/0/semidense_points.csv.gz"
    points, dist_std = model.read_points_file(pt_cloud_path)
    sparse_tensor = model.process_point_cloud(points, dist_std)
    pt_cloud_encoded_features = model(sparse_tensor)
    print("Encoded Features Shape:", pt_cloud_encoded_features.shape)
