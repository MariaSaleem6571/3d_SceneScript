import os
from three_d_scene_script.file_io import read_language_file, process_scene_and_create_zip, export_scene_to_json, create_zip_with_scene, load_scene_from_json_and_obj
from three_d_scene_script.rendering import plot_3d_scene, language_to_bboxes, save_model_images
language_path = "../normalized_gt_scripts/normalized_script_1762.txt"
output_directory = "../meshes"
zip_output_path = "../meshes/scene_bundle.zip"

process_scene_and_create_zip(language_path, output_directory, zip_output_path)

entities = read_language_file(language_path)
original_meshes = language_to_bboxes(entities)
plot_3d_scene(original_meshes, title="Original Scene from Language Commands")

json_file_path = os.path.join(output_directory, "scene.json")
generated_meshes = load_scene_from_json_and_obj(json_file_path, output_directory)
plot_3d_scene(generated_meshes, title="Scene from Generated Meshes")


base_dir = os.path.dirname(os.path.abspath(__file__))
meshes_dir = os.path.join(base_dir, "../meshes")
language_path = os.path.join(base_dir, "generated_scene_script.txt")
zip_output_path = os.path.join(meshes_dir, "scene_bundle.zip")
if not os.path.exists(meshes_dir):
     os.makedirs(meshes_dir)
process_scene_and_create_zip(language_path, meshes_dir, zip_output_path)

entities = read_language_file(language_path)
original_meshes = language_to_bboxes(entities)
plot_3d_scene(original_meshes, title= "Generated Mesh")
save_model_images(original_meshes, output_dir="../images_gt_0")

