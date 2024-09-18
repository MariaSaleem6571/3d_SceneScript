import json
import numpy as np
import plotly.graph_objs as go
import os

json_file_path = '/home/mseleem/3d_SceneScript/scripts/scene.json' 
obj_directory = os.path.dirname(json_file_path)
with open(json_file_path, 'r') as file:
    scene_data = json.load(file)

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

PLOTTING_COLORS = {"wall": "#FBFAF5", "door": "#F7C59F", "window": "#53F4FF"}

def plot_box_wireframe(box, box_class):
    """
    Returns a Plotly scatter trace for a wireframe box.

    Args:
        box: Dictionary with keys "center", "rotation", and "scale".
        box_class: Class of the object (e.g., 'wall', 'door', 'window').
    """


    box_verts = UNIT_CUBE_VERTICES * box["scale"]
    box_verts = (box["rotation"] @ box_verts.T).T + box["center"]
    lines_x, lines_y, lines_z = [], [], []
    for face in UNIT_CUBE_FACES:
        for idx in face:
            lines_x.append(box_verts[idx, 0])
            lines_y.append(box_verts[idx, 1])
            lines_z.append(box_verts[idx, 2])
        lines_x.append(None)
        lines_y.append(None)
        lines_z.append(None)

    return go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z, mode="lines",
        line={"color": PLOTTING_COLORS[box_class], "width": 5},
        name=box_class
    )

# def z_rotation(angle):
#     """Returns a 3x3 rotation matrix for rotation around the z-axis."""
#     s = np.sin(angle)
#     c = np.cos(angle)
#     return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

fig = go.Figure()

for obj in scene_data:
    box_id = obj['id']
    box_class = obj['class']
    position = np.array(obj['pose']['position'])
    rotation_matrix = np.array(obj['pose']['rotation'])
    scale = np.array(obj['scale'])

    box_definition = {
        "center": position,
        "rotation": rotation_matrix,
        "scale": scale
    }

    fig.add_trace(plot_box_wireframe(box_definition, box_class))

fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode="data"
    ),
    title='3D Mesh Scene'
)

fig.show()
