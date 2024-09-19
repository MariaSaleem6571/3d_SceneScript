import os
import json
import zipfile
from three_d_scene_script.rendering import UNIT_CUBE_VERTICES, UNIT_CUBE_FACES, language_to_bboxes
import numpy as np

def read_language_file(filepath):
    """
    Read a language file and return a list of entities.

    Args:
    - filepath (str): path to the language file

    Returns:
    - entities (list): list of entities, where each entity is a tuple (command, entity_parameters)

    """

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

def export_mesh_to_obj(mesh, directory):
    """
    Export a mesh to an OBJ file with the scaling applied directly to the vertices, ensuring thin surfaces for walls, doors, and windows.

    Args:
    - mesh (dict): mesh data
    - directory (str): path to the output directory

    Returns:
    - obj_file_name (str): name of the OBJ file
    """

    numeric_id = ''.join(filter(str.isdigit, mesh['id']))

    obj_file_name = f"{mesh['class']}{numeric_id}.obj"
    obj_file_path = os.path.join(directory, obj_file_name)

    thin_thickness = 0.001  
    if mesh["class"] in {"wall", "door", "window"}:
        mesh["scale"][1] = thin_thickness

    scaled_vertices = UNIT_CUBE_VERTICES * mesh["scale"]
    scaled_vertices = (mesh["rotation"] @ scaled_vertices.T).T + mesh["center"]

    with open(obj_file_path, "w") as obj_file:
        for vertex in scaled_vertices:
            obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in UNIT_CUBE_FACES:
            face_indices = [str(i + 1) for i in face]
            obj_file.write(f"f {' '.join(face_indices)}\n")

    return obj_file_name

def export_scene_to_json(mesh_data, output_directory):
    """
    Export a scene to a JSON file with raw vertices and pose (rotation and position) separately.

    Args:
    - mesh_data (list): list of mesh data
    - output_directory (str): path to the output directory

    Returns:
    - json_file_path (str): path to the JSON file
    """

    scene_data = []

    for mesh in mesh_data:
        thin_thickness = 0.001  
        if mesh["class"] in {"wall", "door", "window"}:
            mesh["scale"][1] = thin_thickness

        raw_vertices = UNIT_CUBE_VERTICES * mesh["scale"]  

        obj_file_name = export_mesh_to_obj(mesh, output_directory)

        object_data = {
            "id": mesh["id"],
            "class": mesh["class"],
            "pose": {
                "position": mesh["center"].tolist(),  
                "rotation": mesh["rotation"].tolist()  
            },
            "vertices": raw_vertices.tolist(), 
            "obj_file": obj_file_name
        }
        scene_data.append(object_data)

    json_file_path = os.path.join(output_directory, "scene.json")
    with open(json_file_path, "w") as json_file:
        json.dump(scene_data, json_file, indent=4)

    return json_file_path

def create_zip_with_scene(output_directory, zip_output_path):
    """
    Create a ZIP file with the scene data.

    Args:
    - output_directory (str): path to the output directory
    - zip_output_path (str): path to the ZIP file

    """

    with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
        for filename in os.listdir(output_directory):
            if filename.endswith('.obj') or filename == 'scene.json':
                file_path = os.path.join(output_directory, filename)
                zipf.write(file_path, filename)

def process_scene_and_create_zip(language_path, output_directory, zip_output_path):
    """
    Process a scene from a language file and create a ZIP file with the scene data.

    Args:
    - language_path (str): path to the language file
    - output_directory (str): path to the output directory
    - zip_output_path (str): path to

    Returns:
    - json_file_path (str): path to the JSON file
    - zip_output_path (str): path to the ZIP file
    """

    entities = read_language_file(language_path)
    boxes = language_to_bboxes(entities)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    export_scene_to_json(boxes, output_directory)
    create_zip_with_scene(output_directory, zip_output_path)

def load_obj_file(obj_file_path):
    """
    Load a mesh from an OBJ file.

    Args:
    - obj_file_path (str): path to the OBJ file

    Returns:
    - vertices (np.array): array of vertices
    - faces (np.array): array of faces

    """
    vertices = []
    faces = []

    with open(obj_file_path, "r") as file:
        for line in file:
            if line.startswith("v "):
                parts = line.strip().split()[1:]
                vertices.append([float(x) for x in parts])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                face = [int(idx.split("/")[0]) - 1 for idx in parts]
                faces.append(face)

    return np.array(vertices), np.array(faces)

def load_scene_from_json_and_obj(json_file_path, obj_directory):
    """
    Load a scene from a JSON file and OBJ files without expecting a 'scale' field.

    Args:
    - json_file_path (str): path to the JSON file
    - obj_directory (str): path to the directory containing the OBJ files

    Returns:
    - mesh_data (list): list of mesh data
    """

    with open(json_file_path, "r") as json_file:
        scene_data = json.load(json_file)

    mesh_data = []

    for obj in scene_data:
        obj_file_name = obj["obj_file"]
        obj_file_path = os.path.join(obj_directory, obj_file_name)
        vertices, faces = load_obj_file(obj_file_path)
        mesh_data.append({
            "id": obj["id"],
            "class": obj["class"],
            "vertices": vertices,  
            "faces": faces,
            "center": np.array(obj["pose"]["position"]),
            "rotation": np.array(obj["pose"]["rotation"]),
        })

    return mesh_data
