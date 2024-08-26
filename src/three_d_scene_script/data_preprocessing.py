import os
from torch.utils.data import Dataset

class PointCloudDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = []

        for scene_dir in os.listdir(root_dir):
            scene_path = os.path.join(root_dir, scene_dir)
            if os.path.isdir(scene_path):
                point_cloud_path = os.path.join(scene_path, 'semidense_points.csv.gz')
                script_path = os.path.join(scene_path, 'ase_scene_language.txt')
                if os.path.exists(point_cloud_path) and os.path.exists(script_path):
                    self.files.append((point_cloud_path, script_path))
                else:
                    print(f"Missing files in directory: {scene_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        point_cloud_path, script_path = self.files[idx]
        return point_cloud_path, script_path




