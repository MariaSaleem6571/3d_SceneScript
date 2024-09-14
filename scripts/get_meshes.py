import os
import numpy as np
import json
import zipfile

UNIT_CUBE_VERTICES = (
    np.array([
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)
    ]) * 0.5
)

UNIT_CUBE_FACES = np.array([
    [0, 1, 3, 2], [4, 5, 7, 6],  
    [0, 2, 6, 4], [1, 3, 7, 5],  
    [0, 1, 5, 4], [2, 3, 7, 6]   
])

def read_language_file(filepath):
    assert os.path.exists(filepath), f"Could not find language file: {filepath}"
    with open(filepath, "r") as f:
        entities = []
        for line in f.readlines():
            line = line.rstrip()
            entries = line.split(", ")
            command = entries[0]
            entity_parameters = {k: float(v) for k, v in [param.split("=") for param in entries[1:]]}
            entities.append((command, entity_parameters))
    return entities

def z_rotation(angle):
    s = np.sin(angle)
    c = np.cos(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def calculate_angle_from_position(corner_a, corner_b):
    direction = corner_b - corner_a
    return np.arctan2(direction[1], direction[0])

def project_point_onto_wall(point, wall_start, wall_end):
    wall_vector = wall_end - wall_start
    point_vector = point - wall_start
    wall_length_squared = np.dot(wall_vector, wall_vector)
    projection_factor = np.dot(point_vector, wall_vector) / wall_length_squared
    projected_point = wall_start + projection_factor * wall_vector
    is_on_wall = 0 <= projection_factor <= 1
    return projected_point, is_on_wall

def find_closest_wall(position, walls):
    min_distance = float('inf')
    closest_wall = None
    closest_angle = None
    projected_position = None

    for wall in walls:
        wall_start = np.array([wall["a_x"], wall["a_y"]])
        wall_end = np.array([wall["b_x"], wall["b_y"]])

        projection, is_on_wall = project_point_onto_wall(position, wall_start, wall_end)
        distance = np.linalg.norm(position - projection)

        if is_on_wall and distance < min_distance:
            min_distance = distance
            closest_wall = wall
            closest_angle = calculate_angle_from_position(wall_start, wall_end)
            projected_position = projection

    return closest_wall, closest_angle, projected_position

def is_angle_within_threshold(angle, wall_angle, threshold=np.pi / 6):
    angle_diff = np.abs(angle - wall_angle)
    return angle_diff <= threshold or np.abs(angle_diff - np.pi) <= threshold

def language_to_meshes(entities):
    mesh_data = []
    walls = []

    for command, params in entities:
        if command == "make_wall":
            wall = {
                "id": int(params["id"]),
                "a_x": params["a_x"],
                "a_y": params["a_y"],
                "b_x": params["b_x"],
                "b_y": params["b_y"],
                "height": params["height"],
                "thickness": params["thickness"]
            }
            walls.append(wall)

            wall_start = np.array([params["a_x"], params["a_y"], params["a_z"]])
            wall_end = np.array([params["b_x"], params["b_y"], params["b_z"]])
            length = np.linalg.norm(wall_start - wall_end)
            angle = calculate_angle_from_position(wall_start, wall_end)
            center = (wall_start + wall_end) * 0.5 + np.array([0, 0, 0.5 * params["height"]])
            scale = np.array([length, params["thickness"], params["height"]])
            rotation = z_rotation(angle)

            transformed_verts = (UNIT_CUBE_VERTICES * scale) @ rotation.T + center

            mesh_data.append({
                "id": f"wall{wall['id']}",
                "cmd": command,
                "class": "wall",
                "vertices": transformed_verts,
                "faces": UNIT_CUBE_FACES,
                "center": center,
                "rotation": rotation
            })

        elif command in {"make_door", "make_window"}:
            identifier = int(params["id"])
            position = np.array([params["position_x"], params["position_y"]])
            closest_wall, closest_angle, projected_position = find_closest_wall(position, walls)

            if closest_wall is None:
                print(f"Warning: Could not find a close wall for {command} at position {position}. Using default angle.")
                closest_angle = 0
                projected_position = position

            angle = calculate_angle_from_position(
                projected_position,
                projected_position + np.array([params["width"], 0])
            )

            if not is_angle_within_threshold(angle, closest_angle):
                print(f"Correcting angle for {command} at position {position} to match wall orientation.")
                angle = closest_angle

            thickness = 0.1
            center = np.array([projected_position[0], projected_position[1], params["position_z"]])
            scale = np.array([params["width"], thickness, params["height"]])
            rotation = z_rotation(angle)

            transformed_verts = (UNIT_CUBE_VERTICES * scale) @ rotation.T + center

            mesh_data.append({
                "id": f"{command[5:]}{identifier}",
                "cmd": command,
                "class": command[5:],  
                "vertices": transformed_verts,
                "faces": UNIT_CUBE_FACES,
                "center": center,
                "rotation": rotation
            })

    return mesh_data

def export_mesh_to_obj(mesh, directory):
    obj_file_name = f"{mesh['id']}.obj"
    obj_file_path = os.path.join(directory, obj_file_name)
    
    with open(obj_file_path, "w") as obj_file:
        for vertex in mesh['vertices']:
            obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in mesh['faces']:
            face_indices = [str(i + 1) for i in face]
            obj_file.write(f"f {' '.join(face_indices)}\n")
    return obj_file_name

def export_scene_to_json(mesh_data, output_directory):
    scene_data = []

    for mesh in mesh_data:
        obj_file_name = export_mesh_to_obj(mesh, output_directory)
        object_data = {
            "id": mesh["id"],
            "class": mesh["class"],
            "pose": {
                "position": mesh["center"].tolist(),
                "rotation": mesh["rotation"].tolist()
            },
            "obj_file": obj_file_name  # Use relative path for the obj file
        }
        scene_data.append(object_data)
    
    json_file_path = os.path.join(output_directory, "scene.json")
    with open(json_file_path, "w") as json_file:
        json.dump(scene_data, json_file, indent=4)
    
    return json_file_path

def create_zip_with_scene(output_directory, zip_output_path):
    with zipfile.ZipFile(zip_output_path, 'w') as zipf:
        for folder_name, subfolders, filenames in os.walk(output_directory):
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                relative_path = os.path.relpath(file_path, output_directory)
                zipf.write(file_path, relative_path)

def process_scene_and_create_zip(language_path, output_directory, zip_output_path):
    entities = read_language_file(language_path)
    meshes = language_to_meshes(entities)
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    export_scene_to_json(meshes, output_directory)
    
    create_zip_with_scene(output_directory, zip_output_path)

file_path = "/home/mseleem/3d_SceneScript/0/ase_scene_language.txt"
output_directory = "/home/mseleem/3d_SceneScript/scripts"
zip_output_path = "/home/mseleem/3d_SceneScript/scripts/scene_bundle.zip"
process_scene_and_create_zip(file_path, output_directory, zip_output_path)
