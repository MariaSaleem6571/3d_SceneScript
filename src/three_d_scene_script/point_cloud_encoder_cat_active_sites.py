import torch
import open3d as o3d
import pandas as pd
import numpy as np
from itertools import product
import os
import torchsparse
from torch import nn
import torchsparse.nn as spnn
import torchsparse.nn.functional as F
import torchsparse.utils as sp_utils
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
import argparse

def read_points_file(filepath):
    assert os.path.exists(filepath), f"Could not find point cloud file: {filepath}"
    df = pd.read_csv(filepath, compression="gzip")
    point_cloud = df[["px_world", "py_world", "pz_world"]]
    dist_std = df["dist_std"]
    return point_cloud.to_numpy(), dist_std.to_numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process voxel size.')
    parser.add_argument('--voxel_size', type=float, default=0.015, help='Voxel size for voxel grid')
    args = parser.parse_args()

    voxel_size = args.voxel_size
    points, dist_std = read_points_file("/home/mseleem/Desktop/3d_model_pt/0/semidense_points.csv.gz")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

    print(f"Number of points: {len(np.asarray(pcd.points))}")
    print("Number of voxels:", len(voxel_grid.get_voxels()))

    o3d.visualization.draw_geometries([voxel_grid])

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

    print(f"Sparse tensor feature dimensions: {sparse_tensor.F.shape}")
    print(f"Sparse tensor coordinate dimensions: {sparse_tensor.C.shape}")

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

    encoder = SparseResNetEncoder().cuda()
    encoded_features = encoder(sparse_tensor.cuda())
    encoded_coordinates = encoded_features.C.float().cuda()
    encoded_coordinates = encoded_coordinates[:, 1:]
    encoded_features_with_pos = torch.cat([encoded_features.F, encoded_coordinates], dim=1)
    coordinates_np = encoded_coordinates.cpu().numpy()
    sorted_indices_np = np.lexsort((coordinates_np[:, 2], coordinates_np[:, 1], coordinates_np[:, 0]))
    sorted_indices = torch.from_numpy(sorted_indices_np).long().cuda()
    sorted_features_with_pos = encoded_features_with_pos[sorted_indices]

    print("Encoded and sorted features with positional encoding:")
    print(sorted_features_with_pos.shape)