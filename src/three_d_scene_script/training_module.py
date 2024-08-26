import os
import torch
import torch.nn as nn
from three_d_scene_script.pt_cloud_encoder import PointCloudTransformerLayer
from three_d_scene_script.gt_processor import SceneScriptProcessor
from three_d_scene_script.decoder_module import Commands, generate_square_subsequent_mask, select_parameters, CommandTransformer
import plotly.graph_objects as go

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_models():
    """
    Initialize the encoder and decoder models

    :return: encoder_model, model
    """
    encoder_model = PointCloudTransformerLayer().to(device)
    model = CommandTransformer(d_model=512, num_layers=6).to(device)
    return encoder_model, model


def prepare_data(encoder_model, pt_cloud_path, scene_script_path):
    """
    Prepare the data for training

    :param encoder_model: The encoder model
    :param pt_cloud_path: The path to the point cloud file
    :param scene_script_path: The path to the scene script file
    :return: pt_cloud_encoded_features, decoder_input_embeddings, gt_output_embeddings
    """

    points, dist_std = encoder_model.read_points_file(pt_cloud_path)
    sparse_tensor = encoder_model.process_point_cloud(points, dist_std)
    pt_cloud_encoded_features = encoder_model(sparse_tensor).to(device)
    processor = SceneScriptProcessor(scene_script_path)
    decoder_input_embeddings, gt_output_embeddings = processor.process()
    return pt_cloud_encoded_features, decoder_input_embeddings.to(device), gt_output_embeddings.to(device)

def configure_model(model, input_dim):
    """
    Configure the model

    :param model: The model to configure
    :param input_dim: The input dimension
    :return: model
    """
    model.set_input_dim(input_dim)
    return model

def initialize_optimizers(model):
    """
    Initialize the optimizer and scheduler

    :param model: The model to optimize
    :return: optimizer, scheduler
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
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

def process_epoch(epoch, num_epochs, model, optimizer, scheduler, pt_cloud_encoded_features, decoder_input_embeddings, gt_output_embeddings):
    """
    Process an epoch

    :param epoch: The epoch number
    :param num_epochs: The total number of epochs
    :param model: The model to train
    :param optimizer: The optimizer
    :param scheduler: The scheduler
    :param pt_cloud_encoded_features: The point cloud encoded features
    :param decoder_input_embeddings: The decoder input embeddings
    :param gt_output_embeddings: The ground truth output embeddings
    :return: total_loss, command_loss, parameter_loss, predictions, ground_truths
    """

    model.train()
    optimizer.zero_grad()

    tgt_mask = generate_square_subsequent_mask(decoder_input_embeddings.size(1)).to(device)
    output = model(src=pt_cloud_encoded_features, tgt=decoder_input_embeddings, tgt_mask=tgt_mask)

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
    Train the model

    :param num_epochs: The number of epochs
    :param model: The model to train
    :param optimizer: The optimizer
    :param scheduler: The scheduler
    :param pt_cloud_encoded_features: The point cloud encoded features
    :param decoder_input_embeddings: The decoder input embeddings
    :param gt_output_embeddings: The ground truth output embeddings
    :return: total_loss_list, command_loss_list, parameter_loss_list, last_epoch_predictions, last_epoch_ground_truths
    """
    total_loss_list, command_loss_list, parameter_loss_list = [], [], []
    last_epoch_predictions, last_epoch_ground_truths = [], []

    for epoch in range(num_epochs):
        total_loss, command_loss, parameter_loss, predictions, ground_truths = process_epoch(
            epoch, num_epochs, model, optimizer, scheduler, pt_cloud_encoded_features, decoder_input_embeddings, gt_output_embeddings
        )

        total_loss_list.append(total_loss)
        command_loss_list.append(command_loss)
        parameter_loss_list.append(parameter_loss)

        if predictions and ground_truths:
            last_epoch_predictions = predictions
            last_epoch_ground_truths = ground_truths

        print(f'Epoch {epoch + 1}/{num_epochs}, Total Loss: {total_loss}, Command Loss: {command_loss}, Parameter Loss: {parameter_loss}')

    return total_loss_list, command_loss_list, parameter_loss_list, last_epoch_predictions, last_epoch_ground_truths

import plotly.graph_objects as go

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

def plot_batch_losses(total_loss_list, command_loss_list, parameter_loss_list):
    """
    Plot the losses over the course of training for each batch.

    :param total_loss_list: The total loss list (can be a list of tensors or numpy arrays)
    :param command_loss_list: The command loss list (can be a list of tensors or numpy arrays)
    :param parameter_loss_list: The parameter loss list (can be a list of tensors or numpy arrays)
    """
    def ensure_list(data):
        if isinstance(data[0], torch.Tensor):
            data = [d.detach().cpu().item() for d in data]
        return data

    total_loss_list = ensure_list(total_loss_list)
    command_loss_list = ensure_list(command_loss_list)
    parameter_loss_list = ensure_list(parameter_loss_list)

    # Use the full range of accumulated losses for the x-axis (each batch is one step)
    x_values = list(range(1, len(total_loss_list) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=total_loss_list, mode='lines', name='Total Loss', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x_values, y=command_loss_list, mode='lines', name='Command Loss', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=x_values, y=parameter_loss_list, mode='lines', name='Parameter Loss', line=dict(color='red')))
    fig.update_layout(
        title='Training Loss over Batches',
        xaxis_title='Batch Index',
        yaxis_title='Loss',
        hovermode='x',
    )
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