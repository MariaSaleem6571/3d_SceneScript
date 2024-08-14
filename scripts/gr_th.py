import pandas as pd
import numpy as np
import torch
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


        if not df_wall.empty:
            df_wall['deltax'] = df_wall['b_x'] - df_wall['a_x']
            df_wall['deltay'] = df_wall['b_y'] - df_wall['a_y']
            df_wall['width'] = np.sqrt(df_wall['deltax']**2 + df_wall['deltay']**2)
            df_wall['theta'] = np.degrees(np.arctan2(df_wall['deltay'], df_wall['deltax']))
            df_wall['xcenter'] = (df_wall['a_x'] + df_wall['b_x']) / 2
            df_wall['ycenter'] = (df_wall['a_y'] + df_wall['b_y']) / 2

        columns_to_drop = ['a_x', 'a_y', 'a_z', 'b_x', 'b_y', 'b_z', 'thickness', 'deltax', 'deltay']
        if not df_wall.empty:
            df_wall = df_wall.drop(columns=[col for col in columns_to_drop if col in df_wall.columns], errors='ignore')
        
        columns_to_drop_door = ['wall0_id', 'wall1_id']
        if not df_door.empty:
            df_door = df_door.drop(columns=[col for col in columns_to_drop_door if col in df_door.columns], errors='ignore')
        
        columns_to_drop_window = ['wall0_id', 'wall1_id']
        if not df_window.empty:
            df_window = df_window.drop(columns=[col for col in columns_to_drop_window if col in df_window.columns], errors='ignore')

        return df_wall, df_door, df_window

    def normalize_dataframe(self, df):
        df_normalized = df.copy()
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for column in numeric_columns:
            if column.startswith('type'):
                continue
        
            min_value = df[column].min()
            max_value = df[column].max()

            if min_value == max_value:
                df_normalized[column] = 0
            else:
                df_normalized[column] = (df[column] - min_value) / (max_value - min_value)

        return df_normalized

    def process(self):
        df_wall, df_door, df_window = self.read_script_to_dataframe()

        df_wall_normalized = self.normalize_dataframe(df_wall)
        df_door_normalized = self.normalize_dataframe(df_door)
        df_window_normalized = self.normalize_dataframe(df_window)

        wall_tensor = torch.tensor(df_wall_normalized.values, dtype=torch.float32)
        door_tensor = torch.tensor(df_door_normalized.values, dtype=torch.float32)
        window_tensor = torch.tensor(df_window_normalized.values, dtype=torch.float32)


        all_data = torch.cat([wall_tensor, door_tensor, window_tensor], dim=0)

        all_tensors = [all_data[i, :].unsqueeze(0) for i in range(all_data.size(0))]


        start_command_vector = Commands.get_one_hot('START')
        stop_command_vector = Commands.get_one_hot('STOP')

        num_parameters = all_data.shape[1] - len(Commands)
        start_parameters = np.zeros(num_parameters)
        stop_parameters = np.zeros(num_parameters)

        start_combined_vector = np.concatenate([start_command_vector, start_parameters])
        stop_combined_vector = np.concatenate([stop_command_vector, stop_parameters])

        start_combined_tensor = torch.tensor(start_combined_vector, dtype=torch.float32)
        stop_combined_tensor = torch.tensor(stop_combined_vector, dtype=torch.float32)

        decoder_input_embeddings_list = [start_combined_tensor] + all_tensors
        gt_output_embeddings_list = all_tensors + [stop_combined_tensor]

        decoder_input_embeddings_list = [tensor.unsqueeze(0) if tensor.dim() == 1 else tensor for tensor in decoder_input_embeddings_list]
        gt_output_embeddings_list = [tensor.unsqueeze(0) if tensor.dim() == 1 else tensor for tensor in gt_output_embeddings_list]

        decoder_input_embeddings = torch.cat(decoder_input_embeddings_list, dim=0)
        gt_output_embeddings = torch.cat(gt_output_embeddings_list, dim=0)

        decoder_input_embeddings = decoder_input_embeddings.unsqueeze(0)
        gt_output_embeddings = gt_output_embeddings.unsqueeze(0)

        return decoder_input_embeddings, gt_output_embeddings

if __name__ == '__main__':
    processor = SceneScriptProcessor('/home/mseleem/Desktop/3d_SceneScript/0/ase_scene_language.txt')
    decoder_input_embeddings, gt_output_embeddings = processor.process()
    print("Shape of decoder_input_embeddings:", decoder_input_embeddings.shape)
    print("Shape of gt_output_embeddings:", gt_output_embeddings.shape)
