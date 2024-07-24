import torch
import open3d as o3d
import pandas as pd
import numpy as np
import os
from torch import nn
import torchsparse
import torchsparse.nn as spnn
from torchsparse import SparseTensor

def read_points_file(filepath):
    assert os.path.exists(filepath), f"Could not find point cloud file: {filepath}"
    df = pd.read_csv(filepath, compression="gzip")
    point_cloud = df[["px_world", "py_world", "pz_world"]]
    dist_std = df["dist_std"]
    print(f"Loaded point cloud with {len(point_cloud)} points.")
    return point_cloud.to_numpy(), dist_std.to_numpy()

def generate_sinusoidal_positional_encoding(coordinates, d_model):
    """
    Generates a sinusoidal positional encoding matrix.
    """
    n_positions, n_dims = coordinates.shape
    pe = torch.zeros(n_positions, d_model).cuda()
    position = coordinates.float().cuda()
    div_term = torch.exp(torch.arange(0, d_model // n_dims, 2).float() * -(np.log(10000.0) / (d_model // n_dims))).cuda()

    for i in range(n_dims):
        pe[:, 2 * i:d_model:2 * n_dims] = torch.sin(position[:, i].unsqueeze(1) * div_term)
        pe[:, 2 * i + 1:d_model:2 * n_dims] = torch.cos(position[:, i].unsqueeze(1) * div_term)

    return pe

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x

class ViTEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.1):
        super(ViTEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.1, emb_dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.context_embedding = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.encoder = ViTEncoder(dim, depth, heads, mlp_dim, dropout)
        # self.norm = nn.LayerNorm(dim)

    def forward(self, x, pos_encoding):
        batch_size = x.shape[0]
        cls_tokens = self.context_embedding.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_encoding = torch.cat((torch.zeros(batch_size, 1, pos_encoding.size(-1)).cuda(), pos_encoding), dim=1)
        x = x + pos_encoding
        x = self.dropout(x)
        x = self.encoder(x)
        # x = self.norm(x)
        return x

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

if __name__ == "__main__":
    voxel_size = 0.03 
    points, dist_std = read_points_file("/home/mseleem/Desktop/3d_model_pt/0/semidense_points.csv.gz")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

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

    voxel_indices_tensor = torch.tensor(filtered_voxel_indices, dtype=torch.int32).cuda()
    features_tensor = torch.tensor(aggregated_features, dtype=torch.float32).view(-1, 1).cuda()

    batch_indices = torch.zeros((voxel_indices_tensor.shape[0], 1), dtype=torch.int32).cuda()
    voxel_indices_tensor_with_batch = torch.cat([batch_indices, voxel_indices_tensor], dim=1)

    sparse_tensor = SparseTensor(features_tensor, voxel_indices_tensor_with_batch)

    encoder = SparseResNetEncoder().cuda()
    encoded_features = encoder(sparse_tensor.cuda())

    print(f"Encoded features (F) from point cloud: {encoded_features.F.shape}")
    print(f"Encoded features (C) from point cloud: {encoded_features.C.shape}")

    positional_encoding = generate_sinusoidal_positional_encoding(encoded_features.C, 512)
    encoded_features_with_pos = encoded_features.F + positional_encoding

    vit = VisionTransformer(dim=512, depth=6, heads=8, mlp_dim=2048).cuda()
    encoded_features_with_pos = encoded_features_with_pos.unsqueeze(0)
    vit_encoded_features = vit(encoded_features_with_pos, positional_encoding.unsqueeze(0))

    vit_encoded_features = vit_encoded_features.squeeze(0)
    print(f"ViT encoded features shape: {vit_encoded_features.shape}")

    # preprocessed_context_embedding = vit_encoded_features[0]  
    # print(f"Preprocessed context embedding shape: {preprocessed_context_embedding.shape}")
