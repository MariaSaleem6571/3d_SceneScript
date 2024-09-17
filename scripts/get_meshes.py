import os
import numpy as np
import json
import zipfile
import plotly.graph_objects as go

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

CLASS_LABELS = {"wall": 0, "door": 1, "window": 2}
PLOTTING_COLORS = {"wall": "#FBFAF5", "door": "#F7C59F", "window": "#53F4FF"}


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

def language_to_bboxes(entities):
    box_definitions = []
    walls = []

    for command, params in entities:
        if command == "make_wall":
            wall = {
                "id": f"wall{int(params['id'])}",
                "class": "wall",
                "a_x": params["a_x"],
                "a_y": params["a_y"],
                "b_x": params["b_x"],
                "b_y": params["b_y"],
                "height": params["height"],
                "thickness": params["thickness"]
            }
            walls.append(wall)

            wall_start = np.array([params["a_x"], params["a_y"], 0])
            wall_end = np.array([params["b_x"], params["b_y"], 0])
            length = np.linalg.norm(wall_start - wall_end)
            angle = calculate_angle_from_position(wall_start, wall_end)
            center = (wall_start + wall_end) * 0.5 + np.array([0, 0, 0.5 * params["height"]])
            scale = np.array([length, params["thickness"], params["height"]])
            rotation = z_rotation(angle)

            box_definitions.append({
                "id": f"wall{wall['id']}",
                "cmd": command,
                "class": "wall",
                "label": CLASS_LABELS["wall"],
                "center": center,
                "rotation": rotation,
                "scale": scale
            })


        elif command in {"make_door", "make_window"}:
            obj_class = "door" if command == "make_door" else "window"
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

            thickness = 0.001
            center = np.array([projected_position[0], projected_position[1], params["position_z"]])
            rotation = z_rotation(angle)
            scale = np.array([params["width"], thickness, params["height"]])

            box_definitions.append({
                "id": f"{obj_class}{identifier}",
                "cmd": command,
                "class": obj_class,
                "label": CLASS_LABELS[obj_class],
                "center": center,
                "rotation": rotation,
                "scale": scale
            })

    return box_definitions


def plot_box_wireframe(box):
    box_verts = UNIT_CUBE_VERTICES * box["scale"]
    box_verts = (box["rotation"] @ box_verts.T).T + box["center"]
    lines_x, lines_y, lines_z = [], [], []

    for pair in UNIT_CUBE_FACES:
        for idx in pair:
            lines_x.append(box_verts[idx, 0])
            lines_y.append(box_verts[idx, 1])
            lines_z.append(box_verts[idx, 2])
        lines_x.append(None)
        lines_y.append(None)
        lines_z.append(None)

    return go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z, name=box["id"], mode="lines",
        line={"color": PLOTTING_COLORS[box["class"]], "width": 10}
    )

def export_mesh_to_obj(mesh, directory):
    numeric_id = ''.join(filter(str.isdigit, mesh['id']))

    obj_file_name = f"{mesh['class']}{numeric_id}.obj"
    obj_file_path = os.path.join(directory, obj_file_name)

    box_verts = UNIT_CUBE_VERTICES * mesh["scale"]
    box_verts = (mesh["rotation"] @ box_verts.T).T + mesh["center"]

    with open(obj_file_path, "w") as obj_file:
        for vertex in box_verts:
            obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in UNIT_CUBE_FACES:
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
            "scale": mesh["scale"].tolist(),
            "obj_file": obj_file_name
        }
        scene_data.append(object_data)

    json_file_path = os.path.join(output_directory, "scene.json")
    with open(json_file_path, "w") as json_file:
        json.dump(scene_data, json_file, indent=4)

    return json_file_path

def create_zip_with_scene(output_directory, zip_output_path):
    with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
        for filename in os.listdir(output_directory):
            if filename.endswith('.obj') or filename == 'scene.json':
                file_path = os.path.join(output_directory, filename)
                zipf.write(file_path, filename)

def process_scene_and_create_zip(language_path, output_directory, zip_output_path):
    entities = read_language_file(language_path)
    boxes = language_to_bboxes(entities)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    export_scene_to_json(boxes, output_directory)
    create_zip_with_scene(output_directory, zip_output_path)

def load_obj_file(obj_file_path):
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
            "scale": np.array(obj["scale"])
        })

    return mesh_data

def plot_3d_scene(mesh_data, title="3D Scene"):
    traces = []

    for mesh in mesh_data:
        box_verts = UNIT_CUBE_VERTICES * mesh["scale"]
        box_verts = (mesh["rotation"] @ box_verts.T).T + mesh["center"]

        faces = UNIT_CUBE_FACES
        poly3d = [[box_verts[vert_id] for vert_id in face] for face in faces]

        lines_x, lines_y, lines_z = [], [], []
        for poly in poly3d:
            for vert in poly:
                lines_x.append(vert[0])
                lines_y.append(vert[1])
                lines_z.append(vert[2])
            lines_x.append(None)
            lines_y.append(None)
            lines_z.append(None)

        object_class = mesh.get("class", "unknown")
        object_color = PLOTTING_COLORS.get(object_class, "#FFFFFF")

        numeric_id = ''.join(filter(str.isdigit, mesh['id']))

        traces.append(go.Scatter3d(
            x=lines_x, y=lines_y, z=lines_z, mode='lines',
            line=dict(color=object_color, width=2),
            name=f"{object_class} ({numeric_id})"
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
            aspectmode='data'
        ),
        template="plotly_dark"
    )
    fig.show()

language_path = "/home/mseleem/3d_SceneScript/scripts/generated_scene_script.txt"
output_directory = "/home/mseleem/3d_SceneScript/scripts"
zip_output_path = "/home/mseleem/3d_SceneScript/scripts/scene_bundle.zip"

process_scene_and_create_zip(language_path, output_directory, zip_output_path)

entities = read_language_file(language_path)
original_meshes = language_to_bboxes(entities)
plot_3d_scene(original_meshes, title="Original Scene from Language Commands")

json_file_path = os.path.join(output_directory, "scene.json")
generated_meshes = load_scene_from_json_and_obj(json_file_path, output_directory)
plot_3d_scene(generated_meshes, title="Scene from Generated Meshes")