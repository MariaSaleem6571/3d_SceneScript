import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from enum import Enum
import torch.nn as nn

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
        self.command_layer = nn.Linear(*transformer_dim, 5)  # Output 5 command logits (START included)
        self.param_layer = nn.Linear(*transformer_dim, 5) # Output 5 params for wall, doors, windows (id not included)

        # self.id_layer = nn.Linear(*transformer_dim, 3)  # Output ids for wall, window, door
        # self.x_layer = nn.Linear(*transformer_dim, 3)  # Output x for wall, window, door
        # self.y_layer = nn.Linear(*transformer_dim, 3)  # Output y for wall, window, door
        # self.z_layer = nn.Linear(*transformer_dim, 2)  # Output z for window, door
        # self.theta_layer = nn.Linear(*transformer_dim, 1)  # Output angle for wall
        # self.w_layer = nn.Linear(*transformer_dim, 3)  # Output width for wall, window, door
        # self.h_layer = nn.Linear(*transformer_dim, 3)  # Output height for wall, window, door

    def forward(self, x):
        # x is the output from the transformer, shape: (batch_size, sequence_length, transformer_dim)
        
        # Predict commands
        command_logits = self.command_layer(x)  # Shape: (batch_size, sequence_length, 5)
        command_probs = F.softmax(command_logits, dim=-1)  # Shape: (batch_size, sequence_length, 5)
        
        parameter_logits = self.param_layer(x) # Shape: (batch_size, sequence_length, 5)
        parameters_probs = F.softmax(parameter_logits, dim=-1) # Shape: (batch_size, sequence_length, 5)
        
        return command_probs, parameters_probs

def select_parameters(command_probs, parameters_probs):
    # command_probs: shape (batch_size, sequence_length, 5)
    # shared_parameters: shape (batch_size, sequence_length, 5)
    # Get the predicted command indices (shape: batch_size, sequence_length)
    command_indx = command_probs.argmax(dim=-1) + 2
    command_indx = command_indx.numpy()
    if command_indx == Commands.STOP.value:
        parameters = torch.zeros(5)
    elif command_indx in [Commands.MAKE_WALL.value, Commands.MAKE_DOOR.value, Commands.MAKE_WINDOW.value]:
        parameters = parameters_probs
    
    return Commands.get_name_for(command_indx), parameters

PARAM_SIZE = 5
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
    if parameters.size(0) < 5:
        parameters = torch.cat((parameters, torch.zeros(5 - parameters.size(0))))

    return torch.cat((one_hot_tensor, parameters))


class CommandTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super(CommandTransformer, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, d_model)
        self.point_cloud_encoder = PointCloudTransformerLayer()
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = CustomTransformerDecoder(d_model, nhead, num_layers, 2048)
        self.output_layer = TransformerOutputLayer((5, d_model))


    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        src_emb = self.point_cloud_encoder(src)
        tgt_emb = self.construct_embedding_vector_from_vocab(command, parameters)  # (seq_len, batch_size, d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        transformer_output = self.transformer(src_emb, tgt_emb)  # (tgt_seq_len, batch_size, d_model)
        
        # command_output = self.fc_command(transformer_output)  # (tgt_seq_len, batch_size, 3)
        ouputs = self.output_layer(transformer_output)  # (tgt_seq_len, batch_size, vocab_size)
        
        return ouputs


model = CommandTransformer(VOCAB_SIZE)
input_emb = construct_embedding_vector_from_vocab(Commands.START, torch.zeros(12)).unsqueeze(-1)
# point_cloud_tensor = torch.zeros((433426,6))

while True:
    pred = model(pt_cloud_tensor, input_emb)
    command, parameters = select_parameters(*pred)
    output_emb = construct_embedding_vector_from_vocab(command, parameters)
    input_emb = torch.cat(input_emb, output_emb.unsqueeze(-1))
    if command == Commands.STOP:
        break

print(input_emb)