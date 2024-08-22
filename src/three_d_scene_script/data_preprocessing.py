import os
import torch
from torch.utils.data import Dataset, DataLoader
from three_d_scene_script.pt_cloud_encoder import PointCloudTransformerLayer
from three_d_scene_script.gt_processor import SceneScriptProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        point_cloud, dist_std = PointCloudTransformerLayer.read_points_file(point_cloud_path)

        script_processor = SceneScriptProcessor(script_path)
        decoder_input, gt_output = script_processor.process()

        return point_cloud, dist_std, decoder_input, gt_output

def custom_collate_fn(batch):
    point_clouds, dist_stds, decoder_inputs, gt_outputs = zip(*batch)

    point_clouds_tensor = torch.tensor(point_clouds[0], dtype=torch.float32, device=device)
    dist_stds_tensor = torch.tensor(dist_stds[0], dtype=torch.float32, device=device)
    decoder_inputs_tensor = decoder_inputs[0].to(device)
    gt_outputs_tensor = gt_outputs[0].to(device)

    return point_clouds_tensor, dist_stds_tensor, decoder_inputs_tensor, gt_outputs_tensor


root_dir = '/home/mseleem/Desktop/projectaria_sandbox/projectaria_tools_ase_data/train'
dataset = PointCloudDataset(root_dir=root_dir)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn, shuffle=True)
transformer_layer = PointCloudTransformerLayer().to(device)

batch_count = 0
for batch in dataloader:
    batch_count += 1
    point_clouds, dist_stds, decoder_inputs, gt_outputs = batch

    sparse_tensor = transformer_layer.process_point_cloud(
        point_clouds.cpu().numpy(), dist_stds.cpu().numpy()
    )
    features = transformer_layer(sparse_tensor).to(device)

    # Debug
    print(f"Batch {batch_count}:")
    print(f"  Features shape: {features.size()}")
    print(f"  Decoder input shape: {decoder_inputs.size()}")
    print(f"  Ground truth output shape: {gt_outputs.size()}")
    print(f"Batch {batch_count} complete.\n")


