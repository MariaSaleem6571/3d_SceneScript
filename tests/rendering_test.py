import json
import numpy as np
import plotly.graph_objs as go
import os

json_file_path = '/home/mseleem/3d_SceneScript/scripts/scene.json' 
obj_directory = os.path.dirname(json_file_path)

with open(json_file_path, 'r') as file:
    scene_data = json.load(file)

PLOTTING_COLORS = {"wall": "#FBFAF5", "door": "#F7C59F", "window": "#53F4FF"}

def parse_obj_file(obj_file_path):
    """
    Parses an OBJ file and extracts vertices and faces.
    
    Args:
        obj_file_path: The path to the OBJ file.
    
    Returns:
        vertices: List of vertices from the OBJ file.
        faces: List of faces from the OBJ file (indices starting from 0).
    """
    vertices = []
    faces = []
    with open(obj_file_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('v '):
                vertex = list(map(float, line.strip().split()[1:]))
                vertices.append(vertex)
            elif line.startswith('f '):
                face = [int(i) - 1 for i in line.strip().split()[1:]]
                faces.append(face)
    
    return np.array(vertices), np.array(faces)

def plot_obj_wireframe(vertices, faces, obj_class):
    """
    Returns a Plotly scatter trace for a wireframe object.
    
    Args:
        vertices: List of vertices for the object.
        faces: List of faces (index triples/quads).
        obj_class: Class of the object (e.g., 'wall', 'door', 'window').
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

fig = go.Figure()

for obj in scene_data:
    obj_id = obj['id']  
    obj_class = obj['class']

    if obj_class == "wall" and obj_id.startswith("wallwall"):
        obj_id = obj_id.replace("wallwall", "wall", 1)
    
    obj_file = os.path.join(obj_directory, f'{obj_id}.obj')  

    if os.path.exists(obj_file):
        vertices, faces = parse_obj_file(obj_file)
        fig.add_trace(plot_obj_wireframe(vertices, faces, obj_class))
    else:
        print(f"OBJ file not found for {obj_id}: {obj_file}")

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
