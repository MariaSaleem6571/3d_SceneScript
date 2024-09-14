import os
import numpy as np
import plotly.graph_objects as go
from collections import Counter

UNIT_CUBE_VERTICES = (
        np.array([
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
            (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)
        ]) * 0.5
)

UNIT_CUBE_LINES_IDXS = np.array([
    [0, 1], [0, 2], [0, 4], [1, 3], [1, 5],
    [2, 3], [2, 6], [3, 7], [4, 5], [4, 6],
    [5, 7], [6, 7]
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
    angle = np.arctan2(direction[1], direction[0])
    return angle

def project_point_onto_wall(point, wall_start, wall_end):
    """
    Projects a point (door/window position) onto a line (wall) defined by start and end points.
    Returns the projected point on the wall and whether the projection is within the wall bounds.
    """
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
            identifier = int(params["id"])

            position = np.array([params["position_x"], params["position_y"]])

            closest_wall, closest_angle, projected_position = find_closest_wall(position, walls)

            if closest_wall is None:
                print(
                    f"Warning: Could not find a close wall for {command} at position {position}. Using default angle.")
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
            rotation = z_rotation(angle)
            scale = np.array([params["width"], thickness, params["height"]])

            box_definitions.append({
                "id": f"{command[5:]}{identifier}",
                "cmd": command,
                "class": command[5:],
                "label": CLASS_LABELS[command[5:]],
                "center": center,
                "rotation": rotation,
                "scale": scale
            })

        else:
            print(f"Entity to box conversion not implemented for: cmd={command}")
            continue

    _compute_counts(box_definitions)
    return box_definitions


def _compute_counts(boxes):
    class_counts = Counter([b["class"] for b in boxes])
    print("Scene contains:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")

def plot_box_wireframe(box):
    box_verts = UNIT_CUBE_VERTICES * box["scale"]
    box_verts = (box["rotation"] @ box_verts.T).T + box["center"]
    lines_x, lines_y, lines_z = [], [], []

    for pair in UNIT_CUBE_LINES_IDXS:
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

def plot_3d_scene(language_path=None):
    traces = []
    if language_path:
        entities = read_language_file(language_path)
        boxes = language_to_bboxes(entities)
        for box in boxes:
            traces.append(plot_box_wireframe(box))

    assert traces, "Nothing to visualize."
    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        scene={"xaxis": {"showticklabels": False, "title": ""},
               "yaxis": {"showticklabels": False, "title": ""},
               "zaxis": {"showticklabels": False, "title": ""}},
    )
    fig.show()

file_path = "/home/mseleem/3d_SceneScript/projectaria_tools_ase_data/test/2005/ase_scene_language.txt"
plot_3d_scene(file_path)
