import os
import torch
import torch.nn as nn
from three_d_scene_script.pt_cloud_encoder import PointCloudTransformerLayer
from three_d_scene_script.gt_processor import SceneScriptProcessor
from three_d_scene_script.decoder_module import Commands, generate_square_subsequent_mask, select_parameters, CommandTransformer
import plotly.graph_objects as go
from torchsparse import SparseTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UnifiedModel(nn.Module):
    def __init__(self, device):
        super(UnifiedModel, self).__init__()
        self.encoder_model = PointCloudTransformerLayer().to(device)
        self.decoder_model = CommandTransformer(d_model=512, num_layers=6).to(device)

    def forward(self, point_cloud, decoder_input_embeddings, tgt_mask):
        if not isinstance(point_cloud, SparseTensor):
            raise ValueError("point_cloud must be a SparseTensor. Ensure you use process_point_cloud.")
        pt_cloud_encoded_features = self.encoder_model(point_cloud)
        output = self.decoder_model(src=pt_cloud_encoded_features, tgt=decoder_input_embeddings, tgt_mask=tgt_mask)
        return output

    def encode(self, point_cloud):
        """
        Separate encoding process for point clouds, ensuring input is a SparseTensor.
        """
        if not isinstance(point_cloud, SparseTensor):
            raise ValueError("point_cloud must be a SparseTensor. Ensure you use process_point_cloud.")
        return self.encoder_model(point_cloud)

    def set_input_dim(self, input_dim):
        """
        Set input dimension for the decoder model.
        """
        self.decoder_model.set_input_dim(input_dim)


def initialize_models(device):
    """
    Initialize and return the unified model.

    :param device: The device to move the model to (e.g., 'cuda' or 'cpu').
    :return: unified_model
    """
    unified_model = UnifiedModel(device)
    return unified_model

def prepare_data(model, pt_cloud_path, scene_script_path, normalize=True):
    """
    Prepare the data for training.

    :param model: The unified model
    :param pt_cloud_path: The path to the point cloud file
    :param scene_script_path: The path to the scene script file
    :return: sparse_tensor, decoder_input_embeddings, gt_output_embeddings
    """

    points, dist_std = model.encoder_model.read_points_file(pt_cloud_path)
    sparse_tensor = model.encoder_model.process_point_cloud(points, dist_std)
    processor = SceneScriptProcessor(scene_script_path)
    processor.set_normalization(normalize)
    decoder_input_embeddings, gt_output_embeddings = processor.process()
    return sparse_tensor, decoder_input_embeddings.to(device), gt_output_embeddings.to(device)

def configure_model(model, input_dim):
    """
    Configure the model

    :param model: The model to configure
    :param input_dim: The input dimension
    :return: model
    """
    model.set_input_dim(input_dim)
    return model

def initialize_optimizers(model, lr=0.0001):
    """
    Initialize the optimizer and learning rate scheduler for the unified model.

    :param model: The unified model containing encoder and decoder
    :param lr: Learning rate for the optimizer
    :return: Initialized optimizer and scheduler
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    return optimizer, scheduler


def calculate_losses(output, gt_output, command_dim, parameter_dim):
    """
    Calculate the losses

    :param output: The output from the model
    :param gt_output: The ground truth output
    :param command_dim: The command dimension
    :param parameter_dim: The parameter dimension
    :return: total_loss, command_loss, parameter_loss
    """
    criterion_command = nn.CrossEntropyLoss()
    criterion_parameters = nn.MSELoss()

    output_command = output[:, :, :command_dim]
    output_parameters = output[:, :, command_dim:]
    gt_command = gt_output[:, :, :command_dim].argmax(dim=-1)
    gt_parameters = gt_output[:, :, command_dim:]

    loss_command = criterion_command(output_command.view(-1, command_dim), gt_command.view(-1))
    loss_parameters = criterion_parameters(output_parameters.view(-1, parameter_dim),
                                           gt_parameters.view(-1, parameter_dim))

    total_loss = loss_command + loss_parameters
    return total_loss, loss_command.item(), loss_parameters.item()

def process_epoch(epoch, num_epochs, model, optimizer, scheduler, point_cloud_sparse_tensor, decoder_input_embeddings, gt_output_embeddings):
    """
    Process an epoch.

    :param epoch: The epoch number
    :param num_epochs: The total number of epochs
    :param model: The unified model
    :param optimizer: The optimizer
    :param scheduler: The scheduler
    :param point_cloud_sparse_tensor: The raw input point cloud in SparseTensor format
    :param decoder_input_embeddings: The decoder input embeddings
    :param gt_output_embeddings: The ground truth output embeddings
    :return: total_loss, command_loss, parameter_loss, predictions, ground_truths
    """

    model.train()
    optimizer.zero_grad()
    tgt_mask = generate_square_subsequent_mask(decoder_input_embeddings.size(1)).to(device)
    output = model(point_cloud_sparse_tensor, decoder_input_embeddings, tgt_mask)

    command_dim = len(Commands)
    parameter_dim = decoder_input_embeddings.size(-1) - command_dim
    total_loss, command_loss, parameter_loss = calculate_losses(output, gt_output_embeddings, command_dim, parameter_dim)

    total_loss.backward()
    optimizer.step()
    scheduler.step()

    predictions, ground_truths = [], []
    if epoch == num_epochs - 1:
        predictions, ground_truths = extract_final_predictions(output, gt_output_embeddings, command_dim)

    return total_loss, command_loss, parameter_loss, predictions, ground_truths


def extract_final_predictions(output, gt_output_embeddings, command_dim):
    """
    Extract the final predictions

    :param output: The output from the model
    :param gt_output_embeddings: The ground truth output embeddings
    :param command_dim: The command dimension
    :return: predictions, ground_truths
    """
    predictions, ground_truths = [], []
    for t in range(output.size(1)):
        selected_command, selected_parameters = select_parameters(output[:, t:t + 1, :command_dim], output[:, t:t + 1, command_dim:])
        predictions.append((selected_command, selected_parameters))
        ground_truths.append((Commands.get_name_for(gt_output_embeddings[:, t, :command_dim].argmax(dim=-1).item() + 1), gt_output_embeddings[:, t, command_dim:]))
    return predictions, ground_truths

def train_model(num_epochs, model, optimizer, scheduler, pt_cloud_encoded_features, decoder_input_embeddings, gt_output_embeddings):
    """
    Train the unified model

    :param num_epochs: The number of epochs
    :param model: The unified model to train
    :param optimizer: The optimizer
    :param scheduler: The scheduler
    :param pt_cloud_encoded_features: The point cloud encoded features
    :param decoder_input_embeddings: The decoder input embeddings
    :param gt_output_embeddings: The ground truth output embeddings
    :return: average_epoch_loss_list, average_epoch_command_loss_list, average_epoch_parameter_loss_list, last_epoch_predictions, last_epoch_ground_truths
    """
    average_epoch_loss_list, average_epoch_command_loss_list, average_epoch_parameter_loss_list = [], [], []
    last_epoch_predictions, last_epoch_ground_truths = [], []

    for epoch in range(num_epochs):
        accumulated_loss = 0.0
        accumulated_command_loss = 0.0
        accumulated_parameter_loss = 0.0
        batch_count = 0

        for pt_cloud, decoder_input, gt_output in zip(pt_cloud_encoded_features, decoder_input_embeddings, gt_output_embeddings):
            total_loss, command_loss, parameter_loss, predictions, ground_truths = process_epoch(
                epoch, num_epochs, model, optimizer, scheduler, pt_cloud, decoder_input, gt_output
            )

            accumulated_loss += total_loss.item()
            accumulated_command_loss += command_loss
            accumulated_parameter_loss += parameter_loss

            batch_count += 1

        average_epoch_loss_list.append(accumulated_loss / batch_count)
        average_epoch_command_loss_list.append(accumulated_command_loss / batch_count)
        average_epoch_parameter_loss_list.append(accumulated_parameter_loss / batch_count)

        print(f'Epoch {epoch + 1}/{num_epochs}, Average Total Loss: {accumulated_loss / batch_count}, '
              f'Average Command Loss: {accumulated_command_loss / batch_count}, '
              f'Average Parameter Loss: {accumulated_parameter_loss / batch_count}')

        if predictions and ground_truths:
            last_epoch_predictions = predictions
            last_epoch_ground_truths = ground_truths

    return average_epoch_loss_list, average_epoch_command_loss_list, average_epoch_parameter_loss_list, last_epoch_predictions, last_epoch_ground_truths

def plot_losses(total_loss_list, command_loss_list, parameter_loss_list, num_epochs):
    """
    Plot the losses

    :param total_loss_list: The total loss list (can be a list of tensors or numpy arrays)
    :param command_loss_list: The command loss list (can be a list of tensors or numpy arrays)
    :param parameter_loss_list: The parameter loss list (can be a list of tensors or numpy arrays)
    :param num_epochs: The number of epochs
    """
    def ensure_list(data):
        if isinstance(data[0], torch.Tensor):
            data = [d.detach().cpu().item() for d in data]
        return data

    total_loss_list = ensure_list(total_loss_list)
    command_loss_list = ensure_list(command_loss_list)
    parameter_loss_list = ensure_list(parameter_loss_list)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(num_epochs)), y=total_loss_list, mode='lines', name='Total Loss', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=list(range(num_epochs)), y=command_loss_list, mode='lines', name='Command Loss', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=list(range(num_epochs)), y=parameter_loss_list, mode='lines', name='Parameter Loss', line=dict(color='red')))
    fig.update_layout(
        title='Training Loss over Epochs',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='x',
    )
    fig.show()

def plot_average_losses(total_loss_list, command_loss_list, parameter_loss_list, save_path=None):
    """
    Plot the average losses per epoch and save the plot to a file.

    :param total_loss_list: The average total loss list (a list of floats)
    :param command_loss_list: The average command loss list (a list of floats)
    :param parameter_loss_list: The average parameter loss list (a list of floats)
    :param save_path: Optional path to save the plot (including filename).
    """
    def ensure_list(data):
        if isinstance(data[0], torch.Tensor):
            data = [d.detach().cpu().item() for d in data]
        return data

    total_loss_list = ensure_list(total_loss_list)
    command_loss_list = ensure_list(command_loss_list)
    parameter_loss_list = ensure_list(parameter_loss_list)

    x_values = list(range(1, len(total_loss_list) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=total_loss_list, mode='lines', name='Total Loss', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x_values, y=command_loss_list, mode='lines', name='Command Loss', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=x_values, y=parameter_loss_list, mode='lines', name='Parameter Loss', line=dict(color='red')))
    fig.update_layout(
        title='Average Training Loss over Epochs',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='x',
    )

    if save_path:
        fig.write_image(save_path)
        print(f"Plot saved at {save_path}")

    fig.show()
def print_last_epoch_results(last_epoch_predictions, last_epoch_ground_truths):
    """
    Print the last epoch results

    :param last_epoch_predictions: The last epoch predictions
    :param last_epoch_ground_truths: The last epoch ground truths
    """
    print("\nPredictions and Ground Truths for the last epoch:")
    for timestep, (pred, gt) in enumerate(zip(last_epoch_predictions, last_epoch_ground_truths)):
        pred_command, pred_parameters = pred
        gt_command, gt_parameters = gt
        pred_parameters = pred_parameters.detach().cpu().numpy()
        gt_parameters = gt_parameters.detach().cpu().numpy()
        print(f"Timestep {timestep + 1}:")
        print(f"Predicted Command: {pred_command}, Predicted Parameters: {pred_parameters}")
        print(f"Ground Truth Command: {gt_command}, Ground Truth Parameters: {gt_parameters}")