import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from enum import Enum

from typing_extensions import List


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


class EmbeddingProcessingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=512):
        super(EmbeddingProcessingNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.fc(x)


class SceneScriptProcessor:
    def __init__(self, file_path, output_dim=512):
        self.file_path = file_path
        self.output_dim = output_dim
        self.embedding_processing_network = None

    def initialize_network(self, input_dim):
        self.embedding_processing_network = EmbeddingProcessingNetwork(input_dim, self.output_dim)

    def parse_line(self, line):
        parts = line.split(',')
        record_type = parts[0].strip()
        one_hot_vector = Commands.get_one_hot(record_type)
        record_dict = {f'type_{i + 1}': val for i, val in enumerate(one_hot_vector)}
        for part in parts[1:]:
            key, value = part.split('=')
            record_dict[key.strip()] = float(value) if '.' in value else int(value)
        return record_dict

    # data preprocessing
    def read_script_to_dataframe(self) -> List[pd.DataFrame]:

        with open(self.file_path, 'r') as file:
            records = [self.parse_line(line.strip()) for line in file if line.strip()]

        df_wall = pd.DataFrame([r for r in records if r['type_3'] == 1])
        df_door = pd.DataFrame([r for r in records if r['type_5'] == 1])
        df_window = pd.DataFrame([r for r in records if r['type_4'] == 1])

        if not df_wall.empty:
            df_wall['deltax'] = df_wall['b_x'] - df_wall['a_x']
            df_wall['deltay'] = df_wall['b_y'] - df_wall['a_y']
            df_wall['width'] = np.sqrt(df_wall['deltax'] ** 2 + df_wall['deltay'] ** 2)
            df_wall['theta'] = np.degrees(np.arctan2(df_wall['deltay'], df_wall['deltax']))
            df_wall['xcenter'] = (df_wall['a_x'] + df_wall['b_x']) / 2
            df_wall['ycenter'] = (df_wall['a_y'] + df_wall['b_y']) / 2

        columns_to_drop = ['a_x', 'a_y', 'a_z', 'b_x', 'b_y', 'b_z', 'thickness', 'deltax', 'deltay']
        if not df_wall.empty:
            df_wall = df_wall.drop(columns=[col for col in columns_to_drop if col in df_wall.columns], errors='ignore')

        columns_to_drop_door = ['wall0_id', 'wall1_id']
        if not df_door.empty:
            df_door = df_door.drop(columns=[col for col in columns_to_drop_door if col in df_door.columns],
                                   errors='ignore')

        columns_to_drop_window = ['wall0_id', 'wall1_id']
        if not df_window.empty:
            df_window = df_window.drop(columns=[col for col in columns_to_drop_window if col in df_window.columns],
                                       errors='ignore')

        # print(len(df_wall.columns), len(df_door.columns), len(df_window.columns)) #DEBUG

        return [df_wall, df_door, df_window]

    def normalize_dataframe(self, df):
        df_normalized = df.copy()
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for column in numeric_columns:
            if column.startswith('type'):
                continue
            df_normalized[column] = (df[column] - df[column].mean()) / df[column].std()

        return df_normalized

    def process_embeddings(self, dataframe):
        input_dim = dataframe.shape[1]
        self.initialize_network(input_dim)
        embeddings = []

        for i in range(len(dataframe)):
            combined_tensor = torch.tensor(dataframe.iloc[i].values, dtype=torch.float32).unsqueeze(0)
            embedding = self.embedding_processing_network(combined_tensor)
            embeddings.append(embedding)

        return embeddings

    def process(self):
        all_data = self.read_script_to_dataframe()
        all_data = self.normalize_and_concat_objects_data(all_data)
        all_embeddings = self.process_embeddings(all_data)
        num_parameters = all_data.shape[1] - len(Commands)
        return (self.get_decoder_input_embeddings(all_embeddings, num_parameters),
                self.get_gt_output_embeddings(all_embeddings, num_parameters))

    def normalize_and_concat_objects_data(self, objects: List[pd.DataFrame]):
        normalized_data = list(map(self.normalize_dataframe, objects))
        return pd.concat(normalized_data, ignore_index=True)

    def get_gt_output_embeddings(self, embeddings, number_of_parameters: int):
        stop_embedding = self.process_stop_embedding(number_of_parameters)
        gt_output_embeddings = embeddings + [stop_embedding]
        gt_output_embeddings = torch.cat(gt_output_embeddings, dim=0).unsqueeze(0)
        return gt_output_embeddings

    def get_decoder_input_embeddings(self, embeddings, number_of_parameters: int):
        start_embedding = self.process_start_embedding(number_of_parameters)
        decoder_input_embeddings = [start_embedding] + embeddings
        decoder_input_embeddings = torch.cat(decoder_input_embeddings, dim=0).unsqueeze(0)
        return decoder_input_embeddings

    def process_start_embedding(self, number_of_parameters: int):
        return self.process_single_command_embedding(Commands.START, np.zeros(number_of_parameters))

    def process_stop_embedding(self, number_of_parameters: int):
        return self.process_single_command_embedding(Commands.STOP, np.zeros(number_of_parameters))

    def process_single_command_embedding(self, command: Commands, parameters: np.ndarray):
        command_vector = Commands.get_one_hot(command.name)
        command_parameters_combined = np.concatenate([command_vector, parameters])
        command_parameters_combined_tensor = torch.tensor(command_parameters_combined, dtype=torch.float32)
        embedding = self.embedding_processing_network(command_parameters_combined_tensor)
        return embedding.unsqueeze(0)