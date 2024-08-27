import os
from torch.utils.data import DataLoader
import torch
from three_d_scene_script.data_preprocessing import PointCloudDataset
from three_d_scene_script.training_module import (
    initialize_models,
    prepare_data,
    configure_model,
    initialize_optimizers,
    process_epoch,
    plot_average_losses
)

root_dir = '/home/mseleem/Desktop/projectaria_sandbox/projectaria_tools_ase_data/train'
base_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
save_dir = os.path.join(base_dir, 'model_checkpoints')
os.makedirs(save_dir, exist_ok=True)

dataset = PointCloudDataset(root_dir=root_dir)
limited_dataset = torch.utils.data.Subset(dataset, list(range(80)))
dataloader = DataLoader(limited_dataset, batch_size=8, shuffle=True)

encoder_model, model = initialize_models()
optimizer, scheduler = None, None
num_epochs = 10

average_epoch_loss_list = []
average_epoch_command_loss_list = []
average_epoch_parameter_loss_list = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    accumulated_loss = 0.0
    accumulated_command_loss = 0.0
    accumulated_parameter_loss = 0.0
    batch_count = 0

    for point_cloud_path, script_path in dataloader:
        batch_count += 1

        pt_cloud_encoded_features, decoder_input_embeddings, gt_output_embeddings = prepare_data(
            encoder_model, point_cloud_path[0], script_path[0]
        )
        if optimizer is None and scheduler is None:
            model = configure_model(model, decoder_input_embeddings.size(-1))
            optimizer, scheduler = initialize_optimizers(model)
        batch_total_loss, batch_command_loss, batch_parameter_loss, _, _ = process_epoch(
            epoch, num_epochs, model, optimizer, scheduler, pt_cloud_encoded_features, decoder_input_embeddings, gt_output_embeddings
        )

        accumulated_loss += batch_total_loss.item()
        accumulated_command_loss += batch_command_loss
        accumulated_parameter_loss += batch_parameter_loss

        print(f"Epoch {epoch + 1}, Batch {batch_count}, Batch Total Loss: {batch_total_loss.item()}, "
              f"Command Loss: {batch_command_loss}, Parameter Loss: {batch_parameter_loss}")

    average_epoch_loss_list.append(accumulated_loss / batch_count)
    average_epoch_command_loss_list.append(accumulated_command_loss / batch_count)
    average_epoch_parameter_loss_list.append(accumulated_parameter_loss / batch_count)

    if (epoch + 1) % 5 == 0:
        model_save_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at {model_save_path}")

    print(f"Epoch {epoch + 1} complete. Average Total Loss: {average_epoch_loss_list[-1]}\n")

plot_average_losses(average_epoch_loss_list, average_epoch_command_loss_list, average_epoch_parameter_loss_list)











