from enum import Enum
import torch.nn as nn
import torch


class Commands(Enum):
    START = 1
    STOP = 2
    MAKE_WALL = 3
    MAKE_WINDOW = 4
    MAKE_DOOR = 5

    @classmethod
    def get_name_for(cls, value: int):
        if value == cls.START.value:
            return cls.START
        if value == cls.STOP.value:
            return cls.STOP
        if value == cls.MAKE_WALL.value:
            return cls.MAKE_WALL
        if value == cls.MAKE_WINDOW.value:
            return cls.MAKE_WINDOW
        if value == cls.MAKE_DOOR.value:
            return cls.MAKE_DOOR

class TransformerOutputLayer(nn.Module):
    def __init__(self, transformer_dim):
        super(TransformerOutputLayer, self).__init__()
        self.command_layer = nn.Linear(*transformer_dim, 4)  # Output 5 command logits
        self.object_id_and_height_layer = nn.Linear(*transformer_dim, 2)  # Output the id and the height of the object
        self.door_or_window_parameters_layer = nn.Linear(*transformer_dim, 6)  # Output the wall_id_0, wall_id_1, x, y, z, w of the door/window
        self.wall_corners_layer = nn.Linear(*transformer_dim, 4)  # Output the x1, y1, x2, y2 of the wall

    def forward(self, x):
        # x is the output from the transformer, shape: (batch_size, sequence_length, transformer_dim)
        
        # Predict commands
        command_logits = self.command_layer(x)  # Shape: (batch_size, sequence_length, 3)
        command_probs = F.softmax(command_logits, dim=-1)  # Shape: (batch_size, sequence_length, 3)
        
        object_id_and_height = self.object_id_and_height_layer(x)  # Shape: (batch_size, sequence_length, 2)

        door_or_window_parameters = self.door_or_window_parameters_layer(x)  # Shape: (batch_size, sequence_length, 6)

        wall_corners = self.wall_corners_layer(x)  # Shape: (batch_size, sequence_length, 4)
        
        return command_probs, object_id_and_height, door_or_window_parameters, wall_corners

def select_parameters(command_probs, object_id_and_height, door_or_window_parameters, wall_corners):
    # command_probs: shape (batch_size, sequence_length, 3)
    # shared_parameter: shape (batch_size, sequence_length, 1)
    # Get the predicted command indices (shape: batch_size, sequence_length)
    command_indx = command_probs.argmax(dim=-1) + 2
    command_indx = command_indx.numpy()
    if command_indx == Commands.STOP.value:
        parameters = torch.zeros(12)
    elif command_indx == Commands.MAKE_WALL.value:
        parameters = torch.cat((object_id_and_height, wall_corners))
    elif command_indx in [Commands.MAKE_DOOR.value, Commands.MAKE_WINDOW.value]:
        parameters = torch.cat((object_id_and_height, door_or_window_parameters))
    
    return Commands.get_name_for(command_indx), parameters

command, parameters = select_parameters(torch.tensor([0, 1, 0, 0 ]), torch.zeros(2), torch.zeros(6), torch.zeros(4)+1)
if parameters.size(0) < 8:
    parameters = torch.cat((parameters, torch.zeros(8 - parameters.size(0))))
print(command, parameters)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from enum import Enum

PARAM_SIZE = 8
vocab = ["START", "STOP", "make_wall", "make_window", "make_door"]
COMMANDS = vocab


vocab_index = [1, 0, 2, 3, 4]
# for command in ["make_wall", "make_window", "make_door"]:
#     # for i in range(PARAM_SIZE):
#     vocab.append(command)

token_to_index = {token: idx for idx, token in enumerate(vocab)}
index_to_token = {idx: token for token, idx in token_to_index.items()}


VOCAB_SIZE = len(vocab) + PARAM_SIZE
print(f"Size of the vocabulary: {VOCAB_SIZE}")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class CustomTransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(CustomTransformerDecoder, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)
        self.d_model = d_model

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # If no encoder is used, we need to provide the memory
        # Normally memory would be the output from the encoder
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return output

# # Example usage
# d_model = 512
# nhead = 8
# num_decoder_layers = 6
# dim_feedforward = 2048
# dropout = 0.1

# decoder = CustomTransformerDecoder(d_model, nhead, num_decoder_layers, dim_feedforward, dropout)

# # Example input tensors
# tgt = torch.rand((10, 32, d_model))  # (sequence_length, batch_size, d_model)
# memory = torch.rand((10, 32, d_model))  # Memory should come from somewhere (e.g., encoder output)

# output = decoder(tgt, memory)
# print(output.shape)


def construct_embedding_vector_from_vocab(command: Commands, parameters: torch.Tensor):
    num_classes = len(Commands)

    # Convert the integer value to a tensor
    value_tensor = torch.tensor(command.value - 1)

    # Create the one-hot tensor
    one_hot_tensor = F.one_hot(value_tensor, num_classes=num_classes)
    if parameters.size(0) < 8:
        parameters = torch.cat((parameters, torch.zeros(8 - parameters.size(0))))

    return torch.cat((one_hot_tensor, parameters))


class CommandTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super(CommandTransformer, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, d_model)
        self.point_cloud_encoder = None #TODO: Add the encoder
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = CustomTransformerDecoder(d_model, nhead, num_layers, 2048)
        self.output_layer = TransformerOutputLayer((12, d_model))


    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        src_emb = self.point_cloud_encoder(src)
        tgt_emb = self.construct_embedding_vector_from_vocab(command, parameters)  # (seq_len, batch_size, d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        transformer_output = self.transformer(src_emb, tgt_emb)  # (tgt_seq_len, batch_size, d_model)
        
        # command_output = self.fc_command(transformer_output)  # (tgt_seq_len, batch_size, 3)
        ouputs = self.output_layer(transformer_output)  # (tgt_seq_len, batch_size, vocab_size)
        
        return ouputs


if __name__ == "__main__":

    model = CommandTransformer(vocab_size=VOCAB_SIZE)
    input_emb = construct_embedding_vector_from_vocab(Commands.START, torch.zeros(12)).unsqueeze(-1)
    point_cloud_tensor = torch.zeros((433426,6))

    while True:
        pred = model(point_cloud_tensor, input_emb)
        command, parameters = select_parameters(*pred)
        output_emb = construct_embedding_vector_from_vocab(command, parameters)
        input_emb = torch.cat(input_emb, output_emb.unsqueeze(-1))
        if command == Commands.STOP:
            break

    print(input_emb)

# def generate_sequence(model):
#     input_seq = torch.tensor([token_to_index["START"]]).unsqueeze(1)  # Start with the START token
#     print(input_seq.shape)
#     tgt_seq = torch.tensor([token_to_index["START"]]).unsqueeze(1)
#     print("START")
#     sequence = []

#     while True:
#         with torch.no_grad():
#             command_output, params_output = model(input_seq, tgt_seq)

#         # Apply softmax to command_output to get probabilities
#         command_probs = F.softmax(command_output[-1, 0], dim=-1)
#         command_index = torch.argmax(command_probs).item()
#         command_token = ["make_wall", "make_window", "make_door", "STOP"][command_index]

#         if command_token == "STOP":
#             print("STOP")
#             sequence.append(command_token)
#             break

#         # Print the predicted command and parameters
#         print(command_token)

#         params = params_output[-1, 0].tolist()
#         print(params)

#         sequence.append((command_token, params))

#         # Update input sequence with the new command
#         new_input = torch.tensor([token_to_index[command_token]]).unsqueeze(1)
#         tgt_seq = torch.cat((tgt_seq, new_input), dim=0)
#         input_seq = torch.cat((input_seq, new_input), dim=0)

#     return sequence

# def parse_predictions(predictions):
#     sequence = ["START"]
#     for command_token, params in predictions:
#         sequence.append(command_token)
#         if command_token != "STOP":
#             sequence.extend(params)
#     return sequence

# # Generate and parse a sequence using the model
# predictions = generate_sequence(model)
# parsed_sequence = parse_predictions(predictions)

# print("Generated Sequence:")
# print(parsed_sequence)
