import torch
from three_d_scene_script.decoder_module import Commands, generate_square_subsequent_mask, \
    construct_embedding_vector_from_vocab, CommandTransformer
from three_d_scene_script.pt_cloud_encoder import PointCloudTransformerLayer
import numpy as np
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UnifiedModel(nn.Module):
    """
    A unified model that combines the point cloud encoder and the command decoder.

    The model takes a point cloud as input and autoregressively generates a scene script.

    The model consists of two parts:
    1. A point cloud encoder that encodes the point cloud into a feature representation.
    2. A command decoder that autoregressively generates a scene script based on the encoded point cloud features.

    The model is trained end-to-end using a transformer-based architecture.

    """

    def __init__(self):
        super(UnifiedModel, self).__init__()
        self.encoder_model = PointCloudTransformerLayer().to(device)
        self.decoder_model = CommandTransformer(input_dim=11, d_model=512, num_layers=6).to(device)

    def forward(self, point_cloud, decoder_input_embeddings, tgt_mask):
        pt_cloud_encoded_features = self.encoder_model(point_cloud)
        output = self.decoder_model(src=pt_cloud_encoded_features, tgt=decoder_input_embeddings, tgt_mask=tgt_mask)
        return output

def prepare_point_cloud(point_cloud_path):
    """
    Read the point cloud from the given file and process it using the point cloud encoder.

    Args:
        point_cloud_path (str): Path to the point cloud file.

    Returns:
        sparse_tensor (torch.sparse.FloatTensor): A sparse tensor representation of the point cloud.
    """

    encoder_model = PointCloudTransformerLayer()
    points, dist_std = encoder_model.read_points_file(point_cloud_path)
    sparse_tensor = encoder_model.process_point_cloud(points, dist_std)
    return sparse_tensor

def convert_wall_prediction(parameters):
    """
    Convert the predicted parameters for a wall into a dictionary.

    Args:
        parameters (np.ndarray): Predicted parameters for a wall.

    Returns:
        dict: A dictionary containing the wall parameters.

    """
    height, width, theta, xcenter, ycenter = parameters
    a_x = xcenter - (width / 2) * np.cos(np.radians(theta))
    a_y = ycenter - (width / 2) * np.sin(np.radians(theta))
    b_x = xcenter + (width / 2) * np.cos(np.radians(theta))
    b_y = ycenter + (width / 2) * np.sin(np.radians(theta))
    return {
        "a_x": a_x,
        "a_y": a_y,
        "a_z": 0.0,
        "b_x": b_x,
        "b_y": b_y,
        "b_z": 0.0,
        "height": height
    }

def convert_door_window_prediction(parameters):
    """
    Convert the predicted parameters for a door or window into a dictionary.

    Args:
        parameters (np.ndarray): Predicted parameters for a door or window.

    Returns:
        dict: A dictionary containing the door or window parameters
    """

    pos_x, pos_y, pos_z, width, height = parameters
    return {
        "position_x": pos_x,
        "position_y": pos_y,
        "position_z": pos_z,
        "width": width,
        "height": height
    }


def generate_and_save_script(model_path, point_cloud_path, max_len=200, output_file="generated_scene_script.txt"):
    """
    Generate a scene script autoregressively and save it to a file using the given model and point cloud.

    Args:
        model_path (str): Path to the trained model.
        point_cloud_path (str): Path to the point cloud file.
        max_len (int): Maximum length of the generated script.
        output_file (str): Path to the output file.

    Returns:
        list: A list of tuples containing the generated script commands and their parameters.
    """

    model = UnifiedModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    point_cloud_sparse_tensor = prepare_point_cloud(point_cloud_path)
    generated_script = []

    command = Commands.START.value - 1
    parameters = torch.zeros(1, 6).to(device)
    decoder_input = construct_embedding_vector_from_vocab(Commands.START, parameters)
    tgt_input = decoder_input.unsqueeze(0)

    wall_count = 0
    door_count = 1000
    window_count = 2000
    script_lines = []

    with torch.no_grad():
        for t in range(max_len):
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            output = model(point_cloud_sparse_tensor, tgt_input, tgt_mask)

            command_probs = output[:, -1, :len(Commands)]
            parameters_probs = output[:, -1, len(Commands):]
            command = command_probs.argmax(dim=-1).item()
            predicted_parameters = parameters_probs.squeeze().cpu().numpy()

            if command + 1 == Commands.STOP.value:
                break

            generated_script.append((Commands.get_name_for(command + 1), predicted_parameters))
            new_input = construct_embedding_vector_from_vocab(Commands.get_name_for(command + 1), parameters_probs)
            tgt_input = torch.cat([tgt_input, new_input.unsqueeze(0)], dim=1)
            parameters = predicted_parameters[1:]

            if Commands.get_name_for(command + 1) == Commands.MAKE_WALL:
                obj_id = wall_count
                wall_count += 1
                wall_data = convert_wall_prediction(parameters)
                script_lines.append(
                    f"make_wall, id={obj_id}, a_x={wall_data['a_x']}, a_y={wall_data['a_y']}, a_z=0.0, "
                    f"b_x={wall_data['b_x']}, b_y={wall_data['b_y']}, b_z=0.0, height={wall_data['height']}, thickness=0.0"
                )

            elif Commands.get_name_for(command + 1) == Commands.MAKE_DOOR:
                obj_id = door_count
                door_count += 1
                door_data = convert_door_window_prediction(parameters)
                script_lines.append(
                    f"make_door, id={obj_id}, position_x={door_data['position_x']}, position_y={door_data['position_y']}, "
                    f"position_z={door_data['position_z']}, width={door_data['width']}, height={door_data['height']}"
                )

            elif Commands.get_name_for(command + 1) == Commands.MAKE_WINDOW:
                obj_id = window_count
                window_count += 1
                window_data = convert_door_window_prediction(parameters)
                script_lines.append(
                    f"make_window, id={obj_id}, position_x={window_data['position_x']}, position_y={window_data['position_y']}, "
                    f"position_z={window_data['position_z']}, width={window_data['width']}, height={window_data['height']}"
                )
    with open(output_file, "w") as f:
        for line in script_lines:
            f.write(line + "\n")

    print(f"Script saved to {output_file}")

    return generated_script


def process_generated_script(predicted_script, output_file="generated_scene_script.txt"):
    """
    Process the generated script and save it to a file.

    Args:
        predicted_script (list): A list of tuples containing the generated script commands and their parameters.
        output_file (str): Path to the output file.

    """

    wall_count = 0
    door_count = 1000
    window_count = 2000

    script_lines = []

    for step, (command, parameters) in enumerate(predicted_script):
        parameters = parameters[1:]

        if command == Commands.MAKE_WALL:
            obj_id = wall_count
            wall_count += 1
            wall_data = convert_wall_prediction(parameters)
            script_lines.append(
                f"make_wall, id={obj_id}, a_x={wall_data['a_x']}, a_y={wall_data['a_y']}, a_z=0.0, "
                f"b_x={wall_data['b_x']}, b_y={wall_data['b_y']}, b_z=0.0, height={wall_data['height']}, thickness=0.0"
            )

        elif command == Commands.MAKE_DOOR:
            obj_id = door_count
            door_count += 1
            door_data = convert_door_window_prediction(parameters)
            script_lines.append(
                f"make_door, id={obj_id}, position_x={door_data['position_x']}, position_y={door_data['position_y']}, "
                f"position_z={door_data['position_z']}, width={door_data['width']}, height={door_data['height']}"
            )

        elif command == Commands.MAKE_WINDOW:
            obj_id = window_count
            window_count += 1
            window_data = convert_door_window_prediction(parameters)
            script_lines.append(
                f"make_window, id={obj_id}, position_x={window_data['position_x']}, position_y={window_data['position_y']}, "
                f"position_z={window_data['position_z']}, width={window_data['width']}, height={window_data['height']}"
            )

    for line in script_lines:
        print(line)

    with open(output_file, "w") as f:
        for line in script_lines:
            f.write(line + "\n")

    print(f"Script saved to {output_file}")


if __name__ == "__main__":
    point_cloud_path = "/test/0/semidense_points.csv.gz"
    model_path = "../model_checkpoints/experiment_5/experiment_5_epoch_200.pth"
    generate_and_save_script(point_cloud_path, model_path)
