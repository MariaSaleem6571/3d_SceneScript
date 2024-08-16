from three_d_scene_script.training_module import (
    initialize_models,
    prepare_data,
    configure_model,
    initialize_optimizers,
    train_model,
    plot_losses,
    print_last_epoch_results
)
import os

encoder_model, model = initialize_models()
pt_cloud_path = os.path.join(os.getcwd(), '../0/semidense_points.csv.gz')
scene_script_path = os.path.join(os.getcwd(), '../0/ase_scene_language.txt')
pt_cloud_encoded_features, decoder_input_embeddings, gt_output_embeddings = prepare_data(
    encoder_model, pt_cloud_path, scene_script_path)
model = configure_model(model, decoder_input_embeddings.size(-1))
optimizer, scheduler = initialize_optimizers(model)
num_epochs = 200
total_loss_list, command_loss_list, parameter_loss_list, last_epoch_predictions, last_epoch_ground_truths = train_model(
    num_epochs, model, optimizer, scheduler, pt_cloud_encoded_features, decoder_input_embeddings, gt_output_embeddings)
plot_losses(total_loss_list, command_loss_list, parameter_loss_list, num_epochs)
print_last_epoch_results(last_epoch_predictions, last_epoch_ground_truths)