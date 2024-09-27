import json
import numpy as np
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
PLOTTING_COLORS = {"wall": "#FBFAF5", "door": "#F7C59F", "window": "#53F4FF"}

def load_json(json_file_path):
    """
    Load JSON data containing mesh information.
    """
    with open(json_file_path, 'r') as file:
        scene_data = json.load(file)
    return scene_data

def apply_pose(vertices, rotation_matrix, position):
    """
    Apply the pose (rotation and translation) to the vertices.
    """
    rotated_vertices = np.dot(vertices, rotation_matrix.T)

    transformed_vertices = rotated_vertices + position
    
    return transformed_vertices

def plot_box_wireframe(vertices, faces, obj_class):
    """
    Returns a Plotly scatter trace for a wireframe box.
    """
    lines_x, lines_y, lines_z = [], [], []
    for face in faces:
        for idx in face:
            lines_x.append(vertices[idx, 0])
            lines_y.append(vertices[idx, 1])
            lines_z.append(vertices[idx, 2])
        lines_x.append(None)
        lines_y.append(None)
        lines_z.append(None)
    return go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z, mode="lines",
        line={"color": PLOTTING_COLORS[obj_class], "width": 5},
        name=obj_class
    )

def plot_3d_scene(json_data):
    """
    Plot the 3D scene using the data from the JSON file.
    """
    fig = go.Figure()

    for obj in json_data:
        obj_class = obj['class']
        position = np.array(obj['pose']['position'])  
        rotation_matrix = np.array(obj['pose']['rotation'])  
        vertices = np.array(obj['vertices'])  

        transformed_vertices = apply_pose(vertices, rotation_matrix, position)

        faces = UNIT_CUBE_FACES  
        fig.add_trace(plot_box_wireframe(transformed_vertices, faces, obj_class))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode="data"
        ),
        title="3D Scene from JSON Data"
    )
    
    fig.show()

json_file_path = "/home/mseleem/3d_SceneScript/meshes/scene.json" 
scene_data = load_json(json_file_path)
plot_3d_scene(scene_data)
