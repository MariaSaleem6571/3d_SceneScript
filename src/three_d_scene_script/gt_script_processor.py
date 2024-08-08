import pandas as pd
import numpy as np
from enum import Enum

class Commands(Enum):
    START = 1
    STOP = 2
    MAKE_WALL = 3
    MAKE_WINDOW = 4
    MAKE_DOOR = 5

    @classmethod
    def get_one_hot(cls, command_type: str):
        command = cls[command_type.upper()]
        one_hot_vector = np.zeros(len(cls))
        one_hot_vector[command.value - 1] = 1
        return one_hot_vector

class SceneScriptProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def parse_line(self, line):
        parts = line.split(',')
        record_type = parts[0].strip()
        one_hot_vector = Commands.get_one_hot(record_type)
        record_dict = {f'type_{i+1}': val for i, val in enumerate(one_hot_vector)}
        for part in parts[1:]:
            key, value = part.split('=')
            record_dict[key.strip()] = float(value) if '.' in value else int(value)
        return record_dict

    def read_script_to_dataframe(self):
        with open(self.file_path, 'r') as file:
            records = [self.parse_line(line.strip()) for line in file if line.strip()]

        df_wall = pd.DataFrame([r for r in records if r['type_3'] == 1])
        df_door = pd.DataFrame([r for r in records if r['type_5'] == 1])
        df_window = pd.DataFrame([r for r in records if r['type_4'] == 1])

        df_wall['deltax'] = df_wall['b_x'] - df_wall['a_x']
        df_wall['deltay'] = df_wall['b_y'] - df_wall['a_y']
        df_wall['width'] = np.sqrt(df_wall['deltax']**2 + df_wall['deltay']**2)
        df_wall['theta'] = np.degrees(np.arctan2(df_wall['deltay'], df_wall['deltax']))
        df_wall['xcenter'] = (df_wall['a_x'] + df_wall['b_x']) / 2
        df_wall['ycenter'] = (df_wall['a_y'] + df_wall['b_y']) / 2

        columns_order = [f'type_{i+1}' for i in range(len(Commands))] + \
                        ['xcenter', 'ycenter', 'theta', 'width', 'height'] + \
                        [col for col in df_wall.columns if col not in [f'type_{i+1}' for i in range(len(Commands))] + ['xcenter', 'ycenter', 'theta', 'width', 'height']]
        df_wall = df_wall[columns_order]

        df_wall = df_wall.drop(columns=['a_x', 'a_y', 'a_z', 'b_x', 'b_y', 'b_z', 'thickness', 'deltax', 'deltay'])
        df_door = df_door.drop(columns=['wall0_id', 'wall1_id'])
        df_window = df_window.drop(columns=['wall0_id', 'wall1_id'])

        return df_wall, df_door, df_window

    def convert_to_vectors(self, df):
        return df.to_numpy()

    def normalize_dataframe(self, df):
        one_hot_columns = [col for col in df.columns if col.startswith('type_')]
        numeric_columns = [col for col in df.columns if col not in one_hot_columns]
        df_numeric_normalized = df[numeric_columns].div(df[numeric_columns].max(axis=0), axis=1)
        df_normalized = pd.concat([df[one_hot_columns], df_numeric_normalized], axis=1)
        return df_normalized

    def process(self):
        df_wall, df_door, df_window = self.read_script_to_dataframe()

        df_wall_normalized = self.normalize_dataframe(df_wall)
        df_door_normalized = self.normalize_dataframe(df_door)
        df_window_normalized = self.normalize_dataframe(df_window)

        walls_vectors = self.convert_to_vectors(df_wall_normalized).flatten()
        doors_vectors = self.convert_to_vectors(df_door_normalized).flatten()
        windows_vectors = self.convert_to_vectors(df_window_normalized).flatten()

        # Debug: print the shape of each vector
        print("Shape of walls_vectors:", walls_vectors.shape)
        print("Shape of doors_vectors:", doors_vectors.shape)
        print("Shape of windows_vectors:", windows_vectors.shape)

        # Concatenate the flattened vectors and reshape to (1, N*11)
        concatenated_vector = np.concatenate([walls_vectors, doors_vectors, windows_vectors]).reshape(1, -1)

        # print("Normalized Walls DataFrame:")
        # print(df_wall_normalized.to_string(index=False))
        # print("\nNormalized Doors DataFrame:")
        # print(df_door_normalized.to_string(index=False))
        # print("\nNormalized Windows DataFrame:")
        # print(df_window_normalized.to_string(index=False))

        print("Concatenated Vector:")
        print(concatenated_vector.shape)

if __name__ == '__main__':
    processor = SceneScriptProcessor('/home/mseleem/Desktop/3d_SceneScript/0/ase_scene_language.txt')
    processor.process()
