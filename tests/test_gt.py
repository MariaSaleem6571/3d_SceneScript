# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from enum import Enum

# class Commands(Enum):
#     START = 1
#     STOP = 2
#     MAKE_WALL = 3
#     MAKE_WINDOW = 4
#     MAKE_DOOR = 5

#     @classmethod
#     def get_one_hot(cls, command_type: str):
#         command = cls[command_type.upper()]
#         one_hot_vector = np.zeros(len(cls))
#         one_hot_vector[command.value - 1] = 1
#         return one_hot_vector

# class ParameterEmbeddingNetwork(nn.Module):
#     def __init__(self, input_dim, embedding_dim=512):
#         super(ParameterEmbeddingNetwork, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, embedding_dim),
#         )

#     def forward(self, x):
#         return self.fc(x)

# class SceneScriptProcessor:
#     def __init__(self, file_path, embedding_dim=512):
#         self.file_path = file_path
#         self.embedding_dim = embedding_dim
#         self.command_embedding_layer = nn.Embedding(len(Commands), embedding_dim)
#         self.init_embedding_layer()
#         self.parameter_embedding_network = ParameterEmbeddingNetwork(1, embedding_dim)

#     def init_embedding_layer(self):
#         torch.manual_seed(42)
#         with torch.no_grad():
#             self.command_embedding_layer.weight = nn.Parameter(torch.randn(len(Commands), self.embedding_dim))

#     def parse_line(self, line):
#         parts = line.split(',')
#         record_type = parts[0].strip()
#         one_hot_vector = Commands.get_one_hot(record_type)
#         record_dict = {f'type_{i+1}': val for i, val in enumerate(one_hot_vector)}
#         for part in parts[1:]:
#             key, value = part.split('=')
#             record_dict[key.strip()] = float(value) if '.' in value else int(value)
#         return record_dict

#     def read_script_to_dataframe(self):
#         with open(self.file_path, 'r') as file:
#             records = [self.parse_line(line.strip()) for line in file if line.strip()]

#         df_wall = pd.DataFrame([r for r in records if r['type_3'] == 1])
#         df_door = pd.DataFrame([r for r in records if r['type_5'] == 1])
#         df_window = pd.DataFrame([r for r in records if r['type_4'] == 1])

#         df_wall['deltax'] = df_wall['b_x'] - df_wall['a_x']
#         df_wall['deltay'] = df_wall['b_y'] - df_wall['a_y']
#         df_wall['width'] = np.sqrt(df_wall['deltax']**2 + df_wall['deltay']**2)
#         df_wall['theta'] = np.degrees(np.arctan2(df_wall['deltay'], df_wall['deltax']))
#         df_wall['xcenter'] = (df_wall['a_x'] + df_wall['b_x']) / 2
#         df_wall['ycenter'] = (df_wall['a_y'] + df_wall['b_y']) / 2

#         columns_order = [f'type_{i+1}' for i in range(len(Commands))] + \
#                         ['xcenter', 'ycenter', 'theta', 'width', 'height'] + \
#                         [col for col in df_wall.columns if col not in [f'type_{i+1}' for i in range(len(Commands))] + ['xcenter', 'ycenter', 'theta', 'width', 'height']]
#         df_wall = df_wall[columns_order]

#         df_wall = df_wall.drop(columns=['a_x', 'a_y', 'a_z', 'b_x', 'b_y', 'b_z', 'thickness', 'deltax', 'deltay'])
#         df_door = df_door.drop(columns=['wall0_id', 'wall1_id'])
#         df_window = df_window.drop(columns=['wall0_id', 'wall1_id'])

#         return df_wall, df_door, df_window

#     def convert_to_vectors(self, df):
#         return df.to_numpy()

#     def normalize_dataframe(self, df):
#         one_hot_columns = [col for col in df.columns if col.startswith('type_')]
#         numeric_columns = [col for col in df.columns if col not in one_hot_columns]
#         df_numeric_normalized = df[numeric_columns].div(df[numeric_columns].max(axis=0), axis=1)
#         df_normalized = pd.concat([df[one_hot_columns], df_numeric_normalized], axis=1)
#         return df_normalized

#     def process(self):
#         df_wall, df_door, df_window = self.read_script_to_dataframe()

#         df_wall_normalized = self.normalize_dataframe(df_wall)
#         df_door_normalized = self.normalize_dataframe(df_door)
#         df_window_normalized = self.normalize_dataframe(df_window)

#         one_hot_columns = [col for col in df_wall.columns if col.startswith('type_')]
#         parameter_columns_wall = ['xcenter', 'ycenter', 'theta', 'width', 'height'] + \
#                                  [col for col in df_wall.columns if col not in one_hot_columns + ['xcenter', 'ycenter', 'theta', 'width', 'height']]
#         parameter_columns_door_window = [col for col in df_door.columns if col not in one_hot_columns]

#         wall_one_hot_vectors = self.convert_to_vectors(df_wall_normalized[one_hot_columns])
#         wall_parameters = self.convert_to_vectors(df_wall_normalized[parameter_columns_wall])
#         door_one_hot_vectors = self.convert_to_vectors(df_door_normalized[one_hot_columns])
#         door_parameters = self.convert_to_vectors(df_door_normalized[parameter_columns_door_window])
#         window_one_hot_vectors = self.convert_to_vectors(df_window_normalized[one_hot_columns])
#         window_parameters = self.convert_to_vectors(df_window_normalized[parameter_columns_door_window])

#         concatenated_embeddings_list = []

#         for i in range(len(wall_one_hot_vectors)):
#             command_tensor = torch.tensor(wall_one_hot_vectors[i], dtype=torch.long)
#             embedded_command = self.command_embedding_layer(command_tensor).detach().numpy()

#             parameters = wall_parameters[i].flatten().reshape(-1, 1)
#             parameters_tensor = torch.tensor(parameters, dtype=torch.float)
#             embedded_parameters = self.parameter_embedding_network(parameters_tensor).detach().numpy()

#             concatenated_embeddings = np.concatenate([embedded_command, embedded_parameters], axis=0)
#             concatenated_embeddings_list.append(concatenated_embeddings)

#         for i in range(len(door_one_hot_vectors)):
#             command_tensor = torch.tensor(door_one_hot_vectors[i], dtype=torch.long)
#             embedded_command = self.command_embedding_layer(command_tensor).detach().numpy()

#             parameters = door_parameters[i].flatten().reshape(-1, 1)
#             parameters_tensor = torch.tensor(parameters, dtype=torch.float)
#             embedded_parameters = self.parameter_embedding_network(parameters_tensor).detach().numpy()

#             concatenated_embeddings = np.concatenate([embedded_command, embedded_parameters], axis=0)
#             concatenated_embeddings_list.append(concatenated_embeddings)

#         for i in range(len(window_one_hot_vectors)):
#             command_tensor = torch.tensor(window_one_hot_vectors[i], dtype=torch.long)
#             embedded_command = self.command_embedding_layer(command_tensor).detach().numpy()

#             parameters = window_parameters[i].flatten().reshape(-1, 1)
#             parameters_tensor = torch.tensor(parameters, dtype=torch.float)
#             embedded_parameters = self.parameter_embedding_network(parameters_tensor).detach().numpy()

#             concatenated_embeddings = np.concatenate([embedded_command, embedded_parameters], axis=0)
#             concatenated_embeddings_list.append(concatenated_embeddings)

#         final_embeddings = np.concatenate(concatenated_embeddings_list, axis=0).reshape(1, -1, self.embedding_dim)

#         return final_embeddings

# if __name__ == '__main__':
#     processor = SceneScriptProcessor('/home/mseleem/Desktop/3d_SceneScript/0/ase_scene_language.txt')
#     embeddings = processor.process()
#     print("Shape of final_embeddings:", embeddings.shape)


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
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

class ParameterEmbeddingNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim=512):
        super(ParameterEmbeddingNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        return self.fc(x)

class SceneScriptProcessor:
    def __init__(self, file_path, embedding_dim=512):
        self.file_path = file_path
        self.embedding_dim = embedding_dim
        self.command_embedding_layer = nn.Embedding(len(Commands), embedding_dim)
        self.init_embedding_layer()
        self.parameter_embedding_network = ParameterEmbeddingNetwork(1, embedding_dim)

    def init_embedding_layer(self):
        torch.manual_seed(42)
        with torch.no_grad():
            self.command_embedding_layer.weight = nn.Parameter(torch.randn(len(Commands), self.embedding_dim))

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

        print(len(df_wall.columns), len(df_door.columns), len(df_window.columns))

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

        one_hot_columns = [col for col in df_wall.columns if col.startswith('type_')]
        parameter_columns_wall = ['xcenter', 'ycenter', 'theta', 'width', 'height'] + \
                                 [col for col in df_wall.columns if col not in one_hot_columns + ['xcenter', 'ycenter', 'theta', 'width', 'height']]
        parameter_columns_door_window = [col for col in df_door.columns if col not in one_hot_columns]

        wall_one_hot_vectors = self.convert_to_vectors(df_wall_normalized[one_hot_columns])
        wall_parameters = self.convert_to_vectors(df_wall_normalized[parameter_columns_wall])
        door_one_hot_vectors = self.convert_to_vectors(df_door_normalized[one_hot_columns])
        door_parameters = self.convert_to_vectors(df_door_normalized[parameter_columns_door_window])
        window_one_hot_vectors = self.convert_to_vectors(df_window_normalized[one_hot_columns])
        window_parameters = self.convert_to_vectors(df_window_normalized[parameter_columns_door_window])

        concatenated_embeddings_list = []

        # Process wall vectors
        for i in range(len(wall_one_hot_vectors)):
            command_tensor = torch.tensor(wall_one_hot_vectors[i], dtype=torch.long)
            embedded_command = self.command_embedding_layer(command_tensor).detach().numpy()

            parameters = wall_parameters[i].flatten().reshape(-1, 1)
            parameters_tensor = torch.tensor(parameters, dtype=torch.float)
            embedded_parameters = self.parameter_embedding_network(parameters_tensor).detach().numpy()

            concatenated_embeddings = np.concatenate([embedded_command, embedded_parameters], axis=0)
            concatenated_embeddings_list.append(concatenated_embeddings)

        # Process door vectors
        for i in range(len(door_one_hot_vectors)):
            command_tensor = torch.tensor(door_one_hot_vectors[i], dtype=torch.long)
            embedded_command = self.command_embedding_layer(command_tensor).detach().numpy()

            parameters = door_parameters[i].flatten().reshape(-1, 1)
            parameters_tensor = torch.tensor(parameters, dtype=torch.float)
            embedded_parameters = self.parameter_embedding_network(parameters_tensor).detach().numpy()

            concatenated_embeddings = np.concatenate([embedded_command, embedded_parameters], axis=0)
            concatenated_embeddings_list.append(concatenated_embeddings)

        # Process window vectors
        for i in range(len(window_one_hot_vectors)):
            command_tensor = torch.tensor(window_one_hot_vectors[i], dtype=torch.long)
            embedded_command = self.command_embedding_layer(command_tensor).detach().numpy()

            parameters = window_parameters[i].flatten().reshape(-1, 1)
            parameters_tensor = torch.tensor(parameters, dtype=torch.float)
            embedded_parameters = self.parameter_embedding_network(parameters_tensor).detach().numpy()

            concatenated_embeddings = np.concatenate([embedded_command, embedded_parameters], axis=0)
            concatenated_embeddings_list.append(concatenated_embeddings)

        # Add START and STOP vectors
        start_command_vector = Commands.get_one_hot('START')
        stop_command_vector = Commands.get_one_hot('STOP')

        start_parameters = np.zeros((len(parameter_columns_wall), 1))  # Assuming parameters are zero for START
        stop_parameters = np.zeros((len(parameter_columns_wall), 1))  # Assuming parameters are zero for STOP

        # print("START command vector:", start_command_vector)
        # print("STOP command vector:", stop_command_vector)
        # print("START parameter vector:", start_parameters.flatten())
        # print("STOP parameter vector:", stop_parameters.flatten())

        start_command_tensor = torch.tensor(start_command_vector, dtype=torch.long)
        embedded_start_command = self.command_embedding_layer(start_command_tensor).detach().numpy()

        stop_command_tensor = torch.tensor(stop_command_vector, dtype=torch.long)
        embedded_stop_command = self.command_embedding_layer(stop_command_tensor).detach().numpy()

        start_parameters_tensor = torch.tensor(start_parameters, dtype=torch.float)
        embedded_start_parameters = self.parameter_embedding_network(start_parameters_tensor).detach().numpy()

        stop_parameters_tensor = torch.tensor(stop_parameters, dtype=torch.float)
        embedded_stop_parameters = self.parameter_embedding_network(stop_parameters_tensor).detach().numpy()

        start_embeddings = np.concatenate([embedded_start_command, embedded_start_parameters], axis=0)
        stop_embeddings = np.concatenate([embedded_stop_command, embedded_stop_parameters], axis=0)

        concatenated_embeddings_list.insert(0, start_embeddings)
        concatenated_embeddings_list.append(stop_embeddings)

        final_embeddings = np.stack(concatenated_embeddings_list, axis=0).reshape(1, -1, self.embedding_dim)

        return final_embeddings

if __name__ == '__main__':
    processor = SceneScriptProcessor('/home/mseleem/Desktop/3d_SceneScript/0/ase_scene_language.txt')
    embeddings = processor.process()
    print("Shape of final_embeddings:", embeddings.shape)
