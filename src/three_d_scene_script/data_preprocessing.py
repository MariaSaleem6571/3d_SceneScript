import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pt_cloud_encoder import PointCloudTransformerLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PointCloudDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = []

        for scene_dir in os.listdir(root_dir):
            scene_path = os.path.join(root_dir, scene_dir)
            if os.path.isdir(scene_path):
                file_path = os.path.join(scene_path, 'semidense_points.csv.gz')
                if os.path.exists(file_path):
                    self.files.append(file_path)
                else:
                    print(f"File not found: {file_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        point_cloud, dist_std = PointCloudTransformerLayer.read_points_file(file_path)
        return point_cloud, dist_std

def custom_collate_fn(batch):
    point_clouds, dist_stds = zip(*batch)
    point_clouds_tensor = torch.tensor(point_clouds[0], dtype=torch.float32, device=device)
    dist_stds_tensor = torch.tensor(dist_stds[0], dtype=torch.float32, device=device)
    return point_clouds_tensor, dist_stds_tensor

root_dir = '/home/mseleem/Desktop/projectaria_sandbox/projectaria_tools_ase_data/train'
dataset = PointCloudDataset(root_dir=root_dir)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn, shuffle=True)
transformer_layer = PointCloudTransformerLayer().to(device)

for batch in dataloader:
    point_clouds, dist_stds = batch
    sparse_tensor = transformer_layer.process_point_cloud(
        point_clouds.cpu().numpy(), dist_stds.cpu().numpy()
    )
    features = transformer_layer(sparse_tensor).to(device)
    print(f"Features shape: {features.size()}")
