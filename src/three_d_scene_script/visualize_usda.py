import open3d as o3d
import numpy as np
import re

def parse_usda(file_path):
    objects = []
    with open(file_path, 'r') as file:
        current_object = None
        for line in file:
            line = line.strip()
            if line.startswith("def Xform"):
                if current_object:
                    objects.append(current_object)
                current_object = {"name": line.split('"')[1], "transform": None, "points": [], "indices": []}
                print(f"Starting new Xform: {current_object['name']}")
            elif line.startswith("matrix4d xformOp:transform"):
                matrix_values = re.findall(r"\((.*?)\)", line)
                if matrix_values:
                    matrix_values = [float(num) for group in matrix_values for num in group.split(',')]
                    current_object["transform"] = np.array(matrix_values).reshape((4, 4))
                    print(
                        f"Transform matrix for {current_object['name']}:\n{current_object['transform']}")
            elif line.startswith("float3[] points"):
                points = []
                while True:
                    line = next(file).strip()
                    if line == "]":
                        break
                    point = tuple(map(float, line.strip('(),').split(',')))
                    points.append(point)
                current_object["points"] = np.array(points)
                print(f"Points for {current_object['name']}:\n{current_object['points']}")
            elif line.startswith("int[] faceVertexIndices"):
                indices = re.findall(r"\d+", line)
                current_object["indices"] = list(map(int, indices))
                print(f"Face indices for {current_object['name']}:\n{current_object['indices']}")
        if current_object:
            objects.append(current_object)
    return objects

def apply_transform(points, transform):
    if transform is not None:
        points = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed_points = points @ transform.T
        return transformed_points[:, :3]
    return points

def create_mesh(points, indices):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)

    triangles = []
    for i in range(0, len(indices), 4):
        if i + 3 < len(indices):
            triangles.append([indices[i], indices[i + 1], indices[i + 2]])
            triangles.append([indices[i], indices[i + 2], indices[i + 3]])

    if len(triangles) > 0:
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        print(f"Created mesh with {len(points)} points and {len(triangles)} triangles.")
        return mesh
    else:
        print("No valid triangles created from indices.")
        return None

def visualize_usda(file_path):
    geometries = parse_usda(file_path)
    mesh_list = []

    for obj in geometries:
        if len(obj['points']) > 0 and len(obj['indices']) > 0:
            transformed_points = apply_transform(obj['points'], obj['transform'])
            mesh = create_mesh(transformed_points, obj['indices'])
            if mesh:
                mesh_list.append(mesh)

    if mesh_list:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='USDA Visualization', width=800, height=600, left=50, top=50, visible=True)
        for mesh in mesh_list:
            vis.add_geometry(mesh)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        vis.run()
        vis.destroy_window()
    else:
        print("No geometries to visualize.")

if __name__ == "__main__":
    usda_file = "scene.usda"
    visualize_usda(usda_file)



