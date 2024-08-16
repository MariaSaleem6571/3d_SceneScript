import torch
import os
from three_d_scene_script.decoder_module import (
    Commands,
    construct_embedding_vector_from_vocab,
    select_parameters, generate_square_subsequent_mask
)
from three_d_scene_script.training_module import (
    initialize_models,
    prepare_data,
    configure_model
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_file_paths():
    """
    Get the file paths for the point cloud and scene script

    :return: pt_cloud_path, scene_script_path
    """

    cwd = os.getcwd()
    pt_cloud_path = os.path.join(cwd, '../0/semidense_points.csv.gz')
    scene_script_path = os.path.join(cwd, '../0/ase_scene_language.txt')
    return pt_cloud_path, scene_script_path

def inference(model, pt_cloud_encoded_features, max_iterations=150):
    """
    Perform inference

    :param model: The model to use for inference
    :param pt_cloud_encoded_features: The point cloud encoded features
    :param max_iterations: The maximum number of iterations
    :return: accumulated_output
    """

    command_dim = len(Commands)
    input_emb = construct_embedding_vector_from_vocab(Commands.START, torch.zeros(1, 6).cuda()).cuda().unsqueeze(0)
    accumulated_output = input_emb
    for i in range(max_iterations):
        tgt_mask = generate_square_subsequent_mask(accumulated_output.size(1)).to(device)
        final_output = model(pt_cloud_encoded_features, accumulated_output, tgt_mask=tgt_mask)
        output_command = final_output[:, :, :command_dim]
        output_parameters = final_output[:, :, command_dim:]
        command, parameters = select_parameters(output_command, output_parameters)
        print(f"Selected command: {command}, parameters: {parameters}")
        if command == Commands.STOP:
            accumulated_output = torch.cat((accumulated_output, final_output[:, -1:, :]), dim=1)
            break
        accumulated_output = torch.cat((accumulated_output, final_output[:, -1:, :]), dim=1)
        if i + 1 >= max_iterations:
            print("Reached maximum iterations")
            break
    print(f"All predictions shape: {accumulated_output.shape}")
    return accumulated_output


if __name__ == "__main__":
    pt_cloud_path, scene_script_path = get_file_paths()
    encoder_model, model = initialize_models()
    pt_cloud_encoded_features, _, _ = prepare_data(encoder_model, pt_cloud_path, scene_script_path)
    input_dim = construct_embedding_vector_from_vocab(Commands.START, torch.zeros(1, 6).cuda()).size(-1)
    model = configure_model(model, input_dim)
    inference(model, pt_cloud_encoded_features)
