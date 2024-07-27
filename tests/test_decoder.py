import torchsparse
from three_d_scene_script.point_cloud_processor import PointCloudTransformerLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from enum import Enum


# Example usage for reading from a file:
pt_cloud_path = "/home/mseleem/Desktop/3d_model_pt/0/semidense_points.csv.gz"
point_cloud_reader = PointCloudTransformerLayer().cuda()
point_cloud, dist_std = point_cloud_reader.read_points_file(pt_cloud_path)
point_cloud_tensor = point_cloud_reader.process_point_cloud(point_cloud, dist_std)

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
        self.command_layer = nn.Linear(transformer_dim, 5)  # Output 5 command logits (START included)
        self.param_layer = nn.Linear(transformer_dim, 6) # Output 6 params for wall, doors, windows 

    def forward(self, x):
        # x is the output from the transformer, shape: (batch_size, sequence_length, transformer_dim)
        
        # Predict commands
        command_logits = self.command_layer(x)  # Shape: (batch_size, sequence_length, 5)
        command_probs = F.softmax(command_logits, dim=-1)  # Shape: (batch_size, sequence_length, 5)
        
        parameter_logits = self.param_layer(x) # Shape: (batch_size, sequence_length, 6)
        parameters_probs = F.softmax(parameter_logits, dim=-1) # Shape: (batch_size, sequence_length, 6)
        
        return command_probs, parameters_probs

def select_parameters(command_probs, parameters_probs):
    # command_probs: shape (batch_size, sequence_length, 5)
    # parameters_probs: shape (batch_size, sequence_length, 6)
    # Get the predicted command indices (shape: batch_size, sequence_length)
    command_indx = command_probs.argmax(dim=-1).item() + 1
    if command_indx == Commands.STOP.value:
        parameters = torch.zeros(6).cuda()
    elif command_indx in [Commands.MAKE_WALL.value, Commands.MAKE_DOOR.value, Commands.MAKE_WINDOW.value]:
        parameters = parameters_probs.squeeze()
    
    return Commands.get_name_for(command_indx), parameters

PARAM_SIZE = 6
vocab = ["START", "STOP", "make_wall", "make_window", "make_door"]
COMMANDS = vocab

VOCAB_SIZE = len(vocab) + PARAM_SIZE
# print(f"Size of the vocabulary: {VOCAB_SIZE}")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model).cuda()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).cuda()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).cuda()
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
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout).cuda()
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers).cuda()
        self.d_model = d_model

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return output

def construct_embedding_vector_from_vocab(command: Commands, parameters: torch.Tensor, d_model=512):
    num_classes = len(Commands)
    one_hot_tensor = F.one_hot(torch.tensor(command.value - 1).cuda(), num_classes=num_classes)

    if parameters.size(0) < 6:
        parameters = torch.cat((parameters, torch.zeros(6 - parameters.size(0)).cuda()))

    combined_tensor = torch.cat((one_hot_tensor, parameters)).float()

    # Project to the correct dimension
    if combined_tensor.size(0) < d_model:
        combined_tensor = F.pad(combined_tensor, (0, d_model - combined_tensor.size(0)))

    return combined_tensor

class CommandTransformer(nn.Module):
    def __init__(self, vocab_size= VOCAB_SIZE, d_model=512, nhead=8, num_layers=6):
        super(CommandTransformer, self).__init__()
        self.point_cloud_encoder = PointCloudTransformerLayer().cuda()
        self.pos_encoder = PositionalEncoding(d_model).cuda()
        self.transformer = CustomTransformerDecoder(d_model, nhead, num_layers, 2048).cuda()
        self.output_layer = TransformerOutputLayer(d_model).cuda()

    def forward(self, src: torchsparse.SparseTensor, tgt: torch.Tensor):
        src_emb = self.point_cloud_encoder(src).cuda()
        print(src_emb.shape)
        # tgt_emb = self.embedding_layer(tgt).cuda()
        tgt_emb = self.pos_encoder(tgt).cuda()
        transformer_output = self.transformer(src_emb, tgt_emb)  # (tgt_seq_len, batch_size, d_model)
        outputs = self.output_layer(transformer_output)  # (tgt_seq_len, batch_size, vocab_size)
        return outputs

model = CommandTransformer().cuda()
input_emb = construct_embedding_vector_from_vocab(Commands.START, torch.zeros(6).cuda()).unsqueeze(-1).cuda()

while True:
    pred = model(point_cloud_tensor, input_emb)
    command, parameters = select_parameters(*pred)
    output_emb = construct_embedding_vector_from_vocab(command, parameters).cuda()
    input_emb = torch.cat(input_emb, output_emb.unsqueeze(-1)).cuda()

    if command == Commands.STOP:
        break

print(input_emb)