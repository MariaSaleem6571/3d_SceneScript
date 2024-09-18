import numpy as np
import plotly.graph_objects as go
import os

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

def z_rotation(angle):
    """
    Returns a 3x3 rotation matrix for a rotation around the
    z-axis by the given angle in radians.

    Args:
        angle: Angle in radians.

    Returns:
        3x3 rotation matrix.
    """

    s = np.sin(angle)
    c = np.cos(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def calculate_angle_from_position(corner_a, corner_b):
    """
    Returns the angle in radians between the x-axis and the line
    connecting corner_a and corner_b.

    Args:
        corner_a: 2D position of the first corner.
        corner_b: 2D position of the second corner.

    Returns:
        Angle in radians between the x-axis and the line connecting
        corner_a and corner_b.
    """

    direction = corner_b - corner_a
    return np.arctan2(direction[1], direction[0])

def project_point_onto_wall(point, wall_start, wall_end):
    """
    Projects a point onto a wall defined by two endpoints.

    Returns the projected point and a boolean indicating whether
    the projected point lies on the wall segment.

    Args:
        point: 2D position of the point.
        wall_start: 2D position of the wall start point.
        wall_end: 2D position of the wall end point.

    Returns:
        Tuple with the projected point and a boolean indicating
        whether the point lies on the wall
    """

    wall_vector = wall_end - wall_start
    point_vector = point - wall_start
    wall_length_squared = np.dot(wall_vector, wall_vector)

    projection_factor = np.dot(point_vector, wall_vector) / wall_length_squared
    projected_point = wall_start + projection_factor * wall_vector

    is_on_wall = 0 <= projection_factor <= 1
    return projected_point, is_on_wall

def find_closest_wall(position, walls):
    """
    Finds the closest wall to a given position.

    Returns the wall, the angle of the wall, and the projected
    position of the point onto the wall.

    Args:
        position: 2D position of the point.
        walls: List of wall dictionaries.

    Returns:
        Tuple with the closest wall, the angle of the wall, and the
        projected position of the point onto the wall.

    """

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
    """
    Checks if an angle is within a threshold of another angle.

    The threshold is applied in both directions.

    Args:
        angle: Angle to check.
        wall_angle: Reference angle.
        threshold: Maximum difference between the angles.

    Returns:
        Boolean indicating whether the angle is within the threshold.
    """

    angle_diff = np.abs(angle - wall_angle)
    return angle_diff <= threshold or np.abs(angle_diff - np.pi) <= threshold

def language_to_bboxes(entities):
    """
    Converts a list of language commands and parameters to a list of

    bounding box definitions for visualization.

    Args:
        entities: List of tuples with a command string and a dictionary
            of parameters.

    Returns:
        List of dictionaries with keys "id", "cmd", "class", "label",
        "center", "rotation", and "scale".
    """

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
    """
    Returns a Plotly scatter trace for a wireframe box.

    The box is defined by a dictionary with keys "center", "rotation",
    and "scale".

    Args:
        box: Dictionary with keys "center", "rotation", and "scale".
    """

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

def plot_3d_scene(mesh_data, title="3D Scene"):
    """
    Plots a 3D scene from a list of bounding box definitions

    Args:
        mesh_data: List of dictionaries with keys "center", "rotation",
            and "scale".
        title: Title of the plot.
    """

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

def save_model_images(mesh_data, output_dir="images"):
    """
    Capture images of the model from 6 fixed positions and save them.
    """
    camera_positions = {
        "top": {"x": 0, "y": 0, "z": 2},
        "bottom": {"x": 0, "y": 0, "z": -2},
        "side1": {"x": 1, "y": 1, "z": 1.5},
        "side2": {"x": -1, "y": -1, "z": 1.5},
        "side3": {"x": 1, "y": -1, "z": 1.5},
        "side4": {"x": -1, "y": 1, "z": 1.5},
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    traces = []

    for box in mesh_data:
        traces.append(plot_box_wireframe(box))

    for view, position in camera_positions.items():
        fig = go.Figure(data=traces)
        fig.update_layout(
            scene_camera=dict(eye=position),
            scene=dict(
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                zaxis=dict(showticklabels=False),
                aspectmode='data'
            ),
            template="plotly_dark"
        )
        file_path = os.path.join(output_dir, f"model_view_{view}.png")
        fig.write_image(file_path)
        print(f"Saved image: {file_path}")
