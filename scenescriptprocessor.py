import pandas as pd
import numpy as np

class SceneScriptProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def parse_line(self, line):
        parts = line.split(',')
        record_type = parts[0].strip()
        record_dict = {'type': record_type}
        for part in parts[1:]:
            key, value = part.split('=')
            record_dict[key.strip()] = float(value) if '.' in value else int(value)
        return record_dict

    def read_script_to_dataframe(self):
        with open(self.file_path, 'r') as file:
            records = [self.parse_line(line.strip()) for line in file if line.strip()]

        df_wall = pd.DataFrame([r for r in records if r['type'] == 'make_wall'])
        df_door = pd.DataFrame([r for r in records if r['type'] == 'make_door'])
        df_window = pd.DataFrame([r for r in records if r['type'] == 'make_window'])

        df_wall['deltax'] = df_wall['b_x'] - df_wall['a_x']
        df_wall['deltay'] = df_wall['b_y'] - df_wall['a_y']
        df_wall['width'] = np.sqrt(df_wall['deltax']**2 + df_wall['deltay']**2)
        df_wall['theta'] = np.degrees(np.arctan2(df_wall['deltay'], df_wall['deltax']))
        df_wall['xcenter'] = (df_wall['a_x'] + df_wall['b_x']) / 2
        df_wall['ycenter'] = (df_wall['a_y'] + df_wall['b_y']) / 2

        columns_order = ['type', 'xcenter', 'ycenter', 'theta', 'width', 'height'] + \
                        [col for col in df_wall.columns if col not in ['type', 'xcenter', 'ycenter', 'theta', 'width', 'height']]
        df_wall = df_wall[columns_order]

        df_wall = df_wall.drop(columns=['id', 'a_x', 'a_y', 'a_z', 'b_x', 'b_y', 'b_z', 'thickness', 'deltax', 'deltay'])
        df_door = df_door.drop(columns=['id', 'wall0_id', 'wall1_id'])
        df_window = df_window.drop(columns=['id', 'wall0_id', 'wall1_id'])

        return df_wall, df_door, df_window

    def convert_to_vectors(self, df):
        return df.to_numpy()

    def process(self):
        df_wall, df_door, df_window = self.read_script_to_dataframe()

        walls_vectors = self.convert_to_vectors(df_wall)
        doors_vectors = self.convert_to_vectors(df_door)
        windows_vectors = self.convert_to_vectors(df_window)

        print("Walls DataFrame:")
        print(df_wall.to_string(index=False))
        print("\nDoors DataFrame:")
        print(df_door.to_string(index=False))
        print("\nWindows DataFrame:")
        print(df_window.to_string(index=False))

        print("Walls Vectors:")
        print(walls_vectors)
        print("\nDoors Vectors:")
        print(doors_vectors)
        print("\nWindows Vectors:")
        print(windows_vectors)


