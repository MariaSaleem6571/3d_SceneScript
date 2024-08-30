import os
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from three_d_scene_script.data_preprocessing import PointCloudDataset
from three_d_scene_script.training_module import (
    initialize_models,
    prepare_data,
    configure_model,
    initialize_optimizers,
    process_epoch,
    plot_average_losses
)
from three_d_scene_script.experiment_config import experiments

def run_experiment(root_dir, base_save_dir, plot_save_dir, voxel_size, normalize, batch_size, num_epochs, experiment_name):
    """
    Run a single experiment with the specified hyperparameters.

    :param root_dir: Directory containing the dataset
    :param base_save_dir: Base directory to save model checkpoints
    :param plot_save_dir: Base directory to save plots
    :param voxel_size: Voxel size for the point cloud processing
    :param normalize: Boolean indicating whether to normalize the data
    :param batch_size: Batch size for DataLoader
    :param num_epochs: Number of epochs to train
    :param experiment_name: Name of the experiment (used for saving models and plot)
    """
    experiment_save_dir = os.path.join(base_save_dir, experiment_name)
    os.makedirs(experiment_save_dir, exist_ok=True)
    os.makedirs(plot_save_dir, exist_ok=True)

    dataset = PointCloudDataset(root_dir=root_dir)
    subset_indices = list(range(24))
    limited_dataset = Subset(dataset, subset_indices)

    dataloader = DataLoader(limited_dataset, batch_size=batch_size, shuffle=True)
    encoder_model, model = initialize_models()
    encoder_model.set_voxel_size(voxel_size)
    optimizer, scheduler = None, None

    average_epoch_loss_list = []
    average_epoch_command_loss_list = []
    average_epoch_parameter_loss_list = []

    with tqdm(total=num_epochs, desc="Training", unit="epoch") as epoch_pbar:
        for epoch in range(num_epochs):
            print(f"Experiment: {experiment_name} - Epoch {epoch + 1}/{num_epochs}")
            accumulated_loss = 0.0
            accumulated_command_loss = 0.0
            accumulated_parameter_loss = 0.0
            batch_count = 0

            for point_cloud_paths, script_paths in dataloader:
                for point_cloud_path, script_path in zip(point_cloud_paths, script_paths):
                    pt_cloud_encoded_features, decoder_input_embeddings, gt_output_embeddings = prepare_data(
                        encoder_model, point_cloud_path, script_path, normalize=normalize
                    )
                if optimizer is None and scheduler is None:
                    model = configure_model(model, decoder_input_embeddings.size(-1))
                    optimizer, scheduler = initialize_optimizers(model)
                batch_total_loss, batch_command_loss, batch_parameter_loss, _, _ = process_epoch(
                    epoch, num_epochs, model, optimizer, scheduler, pt_cloud_encoded_features, decoder_input_embeddings, gt_output_embeddings
                )
                batch_count += 1
                accumulated_loss += batch_total_loss.item()
                accumulated_command_loss += batch_command_loss
                accumulated_parameter_loss += batch_parameter_loss
                print(f"Epoch {epoch + 1}, Batch {batch_count}, Batch Total Loss: {batch_total_loss.item()}, "
                        f"Command Loss: {batch_command_loss}, Parameter Loss: {batch_parameter_loss}")

            average_epoch_loss_list.append(accumulated_loss / batch_count)
            average_epoch_command_loss_list.append(accumulated_command_loss / batch_count)
            average_epoch_parameter_loss_list.append(accumulated_parameter_loss / batch_count)

            if (epoch + 1) % 5 == 0:
                model_save_path = os.path.join(experiment_save_dir, f"{experiment_name}_epoch_{epoch + 1}.pth")
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved at {model_save_path}")

            print(f"Experiment: {experiment_name} - Epoch {epoch + 1} complete. Average Total Loss: {average_epoch_loss_list[-1]}\n")

            epoch_pbar.update(1)

    plot_save_path = os.path.join(plot_save_dir, f"{experiment_name}_loss_plot.png")
    plot_average_losses(average_epoch_loss_list, average_epoch_command_loss_list, average_epoch_parameter_loss_list, save_path=plot_save_path)

def main():
    root_dir = '/home/mseleem/Desktop/projectaria_sandbox/projectaria_tools_ase_data/train'
    base_save_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'model_checkpoints'))
    plot_save_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'plot'))

    os.makedirs(base_save_dir, exist_ok=True)
    os.makedirs(plot_save_dir, exist_ok=True)

    experiment = experiments[0]
    run_experiment(
        root_dir=root_dir,
        base_save_dir=base_save_dir,
        plot_save_dir=plot_save_dir,
        voxel_size=experiment['voxel_size'],
        normalize=experiment['normalize'],
        batch_size=experiment['batch_size'],
        num_epochs=experiment['num_epochs'],
        experiment_name=experiment['name']
    )

if __name__ == "__main__":
    main()
