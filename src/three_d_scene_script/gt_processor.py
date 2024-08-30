import os
import pandas as pd
import numpy as np
import torch
from enum import Enum
from typing import List, Dict, Tuple

class Commands(Enum):
    START = 1
    STOP = 2
    MAKE_WALL = 3
    MAKE_WINDOW = 4
    MAKE_DOOR = 5

    @classmethod
    def get_one_hot(cls, command_type: str) -> np.ndarray:
        """
        Returns one-hot vector for the given command type

        Args: 
            command_type (str): The command type (e.g., 'START', 'STOP').
        """
        command = cls[command_type.upper()]
        one_hot_vector = np.zeros(len(cls))
        one_hot_vector[command.value - 1] = 1
        return one_hot_vector

class SceneScriptProcessor:
    def __init__(self, file_path: str):
        """
        Initializes the SceneScriptProcessor.

        Args:
            file_path (str): Path to the script file.
        """
        self.file_path = file_path
        self.normalize = False

    def set_normalization(self, normalize: bool):
        """
        Sets the normalization flag.

        Args:
            normalize (bool): Whether to normalize the dataframes or not.
        """
        self.normalize = normalize

    def process(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reads, normalizes (if specified), and converts script data into embeddings.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Decoder input and ground truth output embeddings.
        """
        dataframes = self.read_script_to_dataframe()
        tensors = []

        for i, df in enumerate(dataframes):
            if self.normalize:
                df = self.normalize_dataframe(df)
            tensor = torch.tensor(df.values, dtype=torch.float32)

            if tensor.numel() == 0:
                continue
            tensors.append(tensor)

        if tensors:
            all_data = torch.cat(tensors, dim=0)
        else:
            raise ValueError(f"All tensors are empty for script {self.file_path}. Cannot proceed.")

        num_parameters = all_data.shape[1] - len(Commands)

        return (
            self.prepare_decoder_input_embeddings(self.generate_start_embedding(num_parameters), all_data),
            self.prepare_gt_output_embeddings(all_data, self.generate_stop_embedding(num_parameters))
        )

    def prepare_decoder_input_embeddings(self, start_tensor: torch.Tensor, sequence_data: torch.Tensor) -> torch.Tensor:
        """
        Prepares the decoder input embeddings by concatenating the start tensor with the sequence data.

        Returns:
            torch.Tensor: The decoder input embeddings.
        """
        return torch.cat([start_tensor] + [sequence_data[i, :].unsqueeze(0) for i in range(sequence_data.size(0))], dim=0).unsqueeze(0)

    def prepare_gt_output_embeddings(self, sequence_data: torch.Tensor, stop_tensor: torch.Tensor) -> torch.Tensor:
        """
        Prepares the ground truth output embeddings by concatenating the sequence data with the stop tensor.

        Returns:
            torch.Tensor: The ground truth output embeddings     
        """
        return torch.cat([sequence_data[i, :].unsqueeze(0) for i in range(sequence_data.size(0))] + [stop_tensor], dim=0).unsqueeze(0)

    def generate_start_embedding(self, num_parameters: int) -> torch.Tensor:
        """
        Generates the start embedding tensor.

        Returns:
            torch.Tensor: The start embedding tensor.
        """
        return torch.tensor(
            np.concatenate([Commands.get_one_hot('START'), np.zeros(num_parameters)]),
            dtype=torch.float32
        ).unsqueeze(0)

    def generate_stop_embedding(self, num_parameters: int) -> torch.Tensor:
        """
        Generates the stop embedding tensor.

        Returns:
            torch.Tensor: The stop embedding tensor.
        """
        return torch.tensor(
            np.concatenate([Commands.get_one_hot('STOP'), np.zeros(num_parameters)]),
            dtype=torch.float32
        ).unsqueeze(0)

    def read_script_to_dataframe(self) -> List[pd.DataFrame]:
        """
        Reads the script file and converts it into a list of dataframes.

        Returns:
            List[pd.DataFrame]: List of dataframes.
        """

        records = [self.parse_line(line.strip()) for line in open(self.file_path, 'r') if line.strip()]
        return [
            self.process_wall_dataframe(pd.DataFrame([r for r in records if r['type_3'] == 1])),
            self.drop_unused_columns(pd.DataFrame([r for r in records if r['type_5'] == 1]), ['wall0_id', 'wall1_id']),
            self.drop_unused_columns(pd.DataFrame([r for r in records if r['type_4'] == 1]), ['wall0_id', 'wall1_id'])
        ]

    def process_wall_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the wall dataframe by calculating the width, theta, xcenter, and ycenter.

        Returns:
            pd.DataFrame: The processed wall dataframe.
        """
        if not df.empty:
            df['deltax'], df['deltay'] = df['b_x'] - df['a_x'], df['b_y'] - df['a_y']
            df['width'] = np.sqrt(df['deltax'] ** 2 + df['deltay'] ** 2)
            df['theta'] = np.degrees(np.arctan2(df['deltay'], df['deltax']))
            df['xcenter'], df['ycenter'] = (df['a_x'] + df['b_x']) / 2, (df['a_y'] + df['b_y']) / 2
            return self.drop_unused_columns(df, ['a_x', 'a_y', 'a_z', 'b_x', 'b_y', 'b_z', 'thickness', 'deltax', 'deltay'])
        return df

    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes the dataframe.

        Returns:
            pd.DataFrame: The normalized dataframe.
        """
        numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns if not col.startswith('type')]
        for column in numeric_columns:
            min_val, max_val = df[column].min(), df[column].max()
            df[column] = 0 if min_val == max_val else (df[column] - min_val) / (max_val - min_val)
        return df

    def drop_unused_columns(self, df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
        """
        Drops the unused columns from the dataframe.

        Returns:
            pd.DataFrame: The dataframe with unused columns dropped.
        """
        if not df.empty:
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
        return df

    def parse_line(self, line: str) -> Dict[str, float]:
        """
        Parses a line from the script file.

        Returns:
            Dict[str, float]: The parsed line.
        """
        parts = line.split(',')
        record_type = parts[0].strip()
        one_hot_vector = Commands.get_one_hot(record_type)
        record_dict = {f'type_{i+1}': val for i, val in enumerate(one_hot_vector)}
        for part in parts[1:]:
            key, value = part.split('=')
            record_dict[key.strip()] = float(value) if '.' in value else int(value)
        return record_dict
