import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from three_d_scene_script.pt_cloud_encoder import PointCloudTransformerLayer
from three_d_scene_script.data_preprocessing import PointCloudDataset
from three_d_scene_script.gt_processor import SceneScriptProcessor
from three_d_scene_script.decoder_module import Commands, CommandTransformer, generate_square_subsequent_mask
from three_d_scene_script.experiment_config import experiments
import torch.optim as optim
import plotly.graph_objects as go

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_data(model, pt_cloud_path, scene_script_path, voxel_size, normalize):
    """
    Prepare the data for training.

    :param model: The unified model
    :param pt_cloud_path: The path to the point cloud file
    :param scene_script_path: The path to the scene script file
    :param voxel_size: The voxel size for point cloud processing
    :param normalize: Whether to normalize the data
    :return: sparse_tensor, decoder_input_embeddings, gt_output_embeddings
    """
    model.encoder_model.set_voxel_size(voxel_size)
    points, dist_std = model.encoder_model.read_points_file(pt_cloud_path)
    sparse_tensor = model.encoder_model.process_point_cloud(points, dist_std)
    processor = SceneScriptProcessor(scene_script_path)
    processor.set_normalization(normalize)
    decoder_input_embeddings, gt_output_embeddings = processor.process()
    return sparse_tensor, decoder_input_embeddings.to(device), gt_output_embeddings.to(device)


class UnifiedModel(nn.Module):
    def __init__(self):
        super(UnifiedModel, self).__init__()
        self.encoder_model = PointCloudTransformerLayer().to(device)
        self.decoder_model = CommandTransformer(input_dim=11, d_model=512, num_layers=6).to(device)

    def forward(self, point_cloud, decoder_input_embeddings, tgt_mask):
        pt_cloud_encoded_features = self.encoder_model(point_cloud)
        output = self.decoder_model(src=pt_cloud_encoded_features, tgt=decoder_input_embeddings, tgt_mask=tgt_mask)
        return output


def train_model(root_dir, base_save_dir, plot_save_dir, batch_size, num_epochs, voxel_size, normalize, experiment_name):
    """
    Train the model using the specified configuration.

    :param root_dir: Directory containing the dataset
    :param base_save_dir: Directory to save model checkpoints
    :param plot_save_dir: Directory to save training loss plots
    :param batch_size: Batch size
    :param num_epochs: Number of epochs to train
    :param voxel_size: Voxel size for point cloud processing
    :param normalize: Whether to normalize the data
    :param experiment_name: Name of the experiment
    """
    print(f"Training experiment: {experiment_name}")
    print(f"  Voxel size: {voxel_size}")
    print(f"  Normalize: {normalize}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of epochs: {num_epochs}")

    model = UnifiedModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion_command = nn.CrossEntropyLoss()
    criterion_parameters = nn.MSELoss()

    dataset = PointCloudDataset(root_dir=root_dir)
    subset_indices = list(range(5000))
    subset_dataset = Subset(dataset, subset_indices)
    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

    experiment_save_dir = os.path.join(base_save_dir, experiment_name)
    os.makedirs(experiment_save_dir, exist_ok=True)
    os.makedirs(plot_save_dir, exist_ok=True)

    avg_epoch_loss, avg_command_loss, avg_parameter_loss = [], [], []

    with tqdm(total=num_epochs, desc="Training", unit="epoch") as epoch_pbar:
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            total_loss, total_command_loss, total_param_loss = 0.0, 0.0, 0.0
            batch_count = 0

            model.train()
            for point_cloud_paths, script_paths in dataloader:
                sparse_tensor, decoder_input_embeddings, gt_output_embeddings = prepare_data(
                    model, point_cloud_paths[0], script_paths[0], voxel_size, normalize
                )

                tgt_mask = generate_square_subsequent_mask(decoder_input_embeddings.size(1)).to(device)

                output = model(sparse_tensor, decoder_input_embeddings, tgt_mask)

                command_dim = len(Commands)
                param_dim = decoder_input_embeddings.size(-1) - command_dim

                output_command = output[:, :, :command_dim]
                output_parameters = output[:, :, command_dim:]
                gt_command = gt_output_embeddings[:, :, :command_dim].argmax(dim=-1)
                gt_parameters = gt_output_embeddings[:, :, command_dim:]

                command_loss = criterion_command(output_command.view(-1, command_dim), gt_command.view(-1))
                param_loss = criterion_parameters(output_parameters.view(-1, param_dim),
                                                  gt_parameters.view(-1, param_dim))
                total_loss_batch = command_loss + param_loss

                optimizer.zero_grad()
                total_loss_batch.backward()
                optimizer.step()

                total_loss += total_loss_batch.item()
                total_command_loss += command_loss.item()
                total_param_loss += param_loss.item()
                batch_count += 1

                print(
                    f"Batch {batch_count}, Loss: {total_loss_batch.item()}, Command Loss: {command_loss.item()}, Param Loss: {param_loss.item()}")

            scheduler.step()

            avg_epoch_loss.append(total_loss / batch_count)
            avg_command_loss.append(total_command_loss / batch_count)
            avg_parameter_loss.append(total_param_loss / batch_count)

            print(
                f"Epoch {epoch + 1} completed. Avg Loss: {avg_epoch_loss[-1]}, Command Loss: {avg_command_loss[-1]}, Param Loss: {avg_parameter_loss[-1]}\n")

            if (epoch + 1) % 100 == 0:
                model_save_path = os.path.join(experiment_save_dir, f"{experiment_name}_epoch_{epoch + 1}.pth")
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved at {model_save_path}")

            epoch_pbar.update(1)

    plot_save_path = os.path.join(plot_save_dir, f"{experiment_name}_loss_plot.png")
    plot_losses(avg_epoch_loss, avg_command_loss, avg_parameter_loss, num_epochs, plot_save_path)


def plot_losses(total_loss_list, command_loss_list, parameter_loss_list, num_epochs, save_path=None):
    total_loss_list = [l for l in total_loss_list]
    command_loss_list = [l for l in command_loss_list]
    parameter_loss_list = [l for l in parameter_loss_list]

    x_values = list(range(1, len(total_loss_list) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=total_loss_list, mode='lines', name='Total Loss', line=dict(color='blue')))
    fig.add_trace(
        go.Scatter(x=x_values, y=command_loss_list, mode='lines', name='Command Loss', line=dict(color='green')))
    fig.add_trace(
        go.Scatter(x=x_values, y=parameter_loss_list, mode='lines', name='Parameter Loss', line=dict(color='red')))
    fig.update_layout(title='Average Training Loss over Epochs', xaxis_title='Epoch', yaxis_title='Loss', hovermode='x')

    if save_path:
        fig.write_image(save_path)
        print(f"Plot saved at {save_path}")

    fig.show()


def main():
    root_dir = '../projectaria_tools_ase_data/train'
    base_save_dir = os.path.abspath(os.path.join(os.getcwd(), '../model_checkpoints'))
    plot_save_dir = os.path.abspath(os.path.join(os.getcwd(), '../plot'))

    os.makedirs(base_save_dir, exist_ok=True)
    os.makedirs(plot_save_dir, exist_ok=True)

    experiment = experiments[2]

    train_model(
        root_dir=root_dir,
        base_save_dir=base_save_dir,
        plot_save_dir=plot_save_dir,
        batch_size=experiment['batch_size'],
        num_epochs=experiment['num_epochs'],
        voxel_size=experiment['voxel_size'],
        normalize=experiment['normalize'],
        experiment_name=experiment['name']
    )


if __name__ == "__main__":
    main()
