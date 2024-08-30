import open3d as o3d
import numpy as np
import pandas as pd

def create_lineset_from_box(min_bound, max_bound, color=[0, 0, 0]):
    points = [
        [min_bound[0], min_bound[1], min_bound[2]], [max_bound[0], min_bound[1], min_bound[2]],
        [max_bound[0], max_bound[1], min_bound[2]], [min_bound[0], max_bound[1], min_bound[2]],
        [min_bound[0], min_bound[1], max_bound[2]], [max_bound[0], min_bound[1], max_bound[2]],
        [max_bound[0], max_bound[1], max_bound[2]], [min_bound[0], max_bound[1], max_bound[2]]
    ]
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    colors = [color for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def make_wall(xcenter, ycenter, theta, width, height):
    min_bound = [-width / 2, -0.05, 0]
    max_bound = [width / 2, 0.05, height]
    wall = create_lineset_from_box(min_bound, max_bound, color=[0.8, 0.8, 0.8])
    R = wall.get_rotation_matrix_from_xyz([0, 0, np.deg2rad(theta)])
    wall.rotate(R, center=[0, 0, 0])
    wall.translate([xcenter, ycenter, height / 2])
    return wall

def make_door(position_x, position_y, position_z, width, height, rotation):
    min_bound = [-width / 2, -0.05, 0]
    max_bound = [width / 2, 0.05, height]
    door = create_lineset_from_box(min_bound, max_bound, color=[0.5, 0.3, 0.2])
    R = door.get_rotation_matrix_from_xyz([0, 0, np.deg2rad(rotation)])
    door.rotate(R, center=[0, 0, 0])
    door.translate([position_x, position_y, position_z])
    return door

def make_window(position_x, position_y, position_z, width, height, rotation):
    min_bound = [-width / 2, -0.05, 0]
    max_bound = [width / 2, 0.05, height]
    window = create_lineset_from_box(min_bound, max_bound, color=[0.2, 0.5, 0.7])
    R = window.get_rotation_matrix_from_xyz([0, 0, np.deg2rad(rotation)])
    window.rotate(R, center=[0, 0, 0])
    window.translate([position_x, position_y, position_z])
    return window

# Walls data
walls_df = pd.DataFrame({
    'type': ['make_wall'] * 8,
    'xcenter': [1.251978, 5.069176, 1.251978, -2.565221, -0.100031, 2.371447, -0.100031, -2.571509],
    'ycenter': [6.164665, 4.038439, 1.912213, 4.038439, 1.766310, -0.764045, -3.294399, -0.764045],
    'theta': [0.0, -90.0, 180.0, 90.0, 0.0, -90.0, 180.0, 90.0],
    'width': [7.634396, 4.252452, 7.634396, 4.252452, 4.942956, 5.060709, 4.942956, 5.060709],
    'height': [3.26243] * 8
})

# Doors data
doors_df = pd.DataFrame({
    'type': ['make_door'] * 3,
    'position_x': [-1.511862, 2.870784, 5.069176],
    'position_y': [1.839261, 6.164665, 3.399479],
    'position_z': [1.011814, 0.993711, 0.983353],
    'width': [1.820626, 1.690708, 1.788263],
    'height': [2.023629, 1.987422, 1.966706],
    'rotation': [0.0, -90.0, -90.0]
})

# Windows data
windows_df = pd.DataFrame({
    'type': ['make_window'] * 4,
    'position_x': [4.447797, -2.565221, -1.626487, -2.571509],
    'position_y': [6.164665, 3.221880, -3.294399, -0.303761],
    'position_z': [1.644804, 1.375980, 2.249849, 1.696074],
    'width': [1.007971, 2.342133, 1.034733, 3.890121],
    'height': [2.118911, 2.336972, 1.210511, 2.829068],
    'rotation': [0.0, 90.0, 180.0, 90.0]
})

walls = []
for index, row in walls_df.iterrows():
    walls.append(make_wall(row['xcenter'], row['ycenter'], row['theta'], row['width'], row['height']))

doors = []
for index, row in doors_df.iterrows():
    doors.append(make_door(row['position_x'], row['position_y'], row['position_z'], row['width'], row['height'], row['rotation']))

windows = []
for index, row in windows_df.iterrows():
    windows.append(make_window(row['position_x'], row['position_y'], row['position_z'], row['width'], row['height'], row['rotation']))

scene = walls + doors + windows

o3d.visualization.draw_geometries(scene)
