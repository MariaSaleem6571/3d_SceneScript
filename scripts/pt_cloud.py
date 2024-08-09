import torch
from three_d_scene_script.point_cloud_sinusoidal_pt_cloud_no_vit import PointCloudTransformerLayer
import torch.nn as nn
import torch.nn.functional as F
import math
from enum import Enum

encoder_model = PointCloudTransformerLayer().cuda()
pt_cloud_path = "/home/mseleem/Desktop/3d_model_pt/0/semidense_points.csv.gz"
points, dist_std = encoder_model.read_points_file(pt_cloud_path)
sparse_tensor = encoder_model.process_point_cloud(points, dist_std)
pt_cloud_encoded_features = encoder_model(sparse_tensor)

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
        self.command_layer = nn.Linear(transformer_dim, 5)  
        self.param_layer = nn.Linear(transformer_dim, 6)  # Output 6 params for wall, doors, windows 

    def forward(self, x):
        command_logits = self.command_layer(x)  # Shape: (batch_size, sequence_length, 5)

        # Mask the "START" command logits to be very negative
        command_logits[..., 0] = -float('inf') 

        command_probs = F.softmax(command_logits, dim=-1)  # Shape: (batch_size, sequence_length, 5)
        parameter_logits = self.param_layer(x)  # Shape: (batch_size, sequence_length, 6)
        parameters_probs = torch.tanh(parameter_logits)  # Shape: (batch_size, sequence_length, 6) # tanh ko kese lao gi wapis raw logits mein ispe socho
        return command_probs, parameters_probs

def select_parameters(command_probs, parameters_probs):
    command_indx = command_probs[0, 0].argmax(dim=-1).item() + 1 
    if command_indx == Commands.STOP.value:
        parameters = torch.zeros(6).cuda()
    elif command_indx in [Commands.MAKE_WALL.value, Commands.MAKE_DOOR.value, Commands.MAKE_WINDOW.value]:
        parameters = parameters_probs.squeeze()
    else:
        parameters = torch.zeros(6).cuda()
    return Commands.get_name_for(command_indx), parameters

PARAM_SIZE = 6
vocab = ["START", "STOP", "make_wall", "make_window", "make_door"]
COMMANDS = vocab

VOCAB_SIZE = len(vocab) + PARAM_SIZE

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

class CrossAttention(nn.Module):
    def __init__(self, d_model, d_out_kq, d_out_v):
        super(CrossAttention, self).__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Linear(d_model, d_out_kq)
        self.W_key = nn.Linear(d_model, d_out_kq)
        self.W_value = nn.Linear(d_model, d_out_v)
    
    def forward(self, x_1, x_2):
        queries_1 = self.W_query(x_1)
        keys_2 = self.W_key(x_2)
        values_2 = self.W_value(x_2)
        
        attn_scores = queries_1.matmul(keys_2.transpose(-2, -1))
        attn_weights = torch.softmax(attn_scores / self.d_out_kq**0.5, dim=-1)
        
        context_vec = attn_weights.matmul(values_2)
        return context_vec


class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, d_out_kq, d_out_v, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, 1, dropout=dropout) # Self-attention: tgt, tgt, tgt
        self.cross_attention = CrossAttention(d_model, d_out_kq, d_out_v)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self-attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention
        tgt2 = self.cross_attention(tgt, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feedforward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class CustomTransformerDecoder(nn.Module):
    def __init__(self, d_model, d_out_kq, d_out_v, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(CustomTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            CustomTransformerDecoderLayer(d_model, d_out_kq, d_out_v, dim_feedforward, dropout) for _ in range(num_decoder_layers)
        ])

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return tgt

def construct_embedding_vector_from_vocab(command: Commands, parameters: torch.Tensor):
    num_classes = len(Commands)
    value_tensor = torch.tensor(command.value - 1).cuda()
    one_hot_tensor = F.one_hot(value_tensor, num_classes=num_classes).float().cuda()

    # Ensure one_hot_tensor is 2D by adding a batch dimension if necessary
    if one_hot_tensor.dim() == 1:
        one_hot_tensor = one_hot_tensor.unsqueeze(0)  # Shape: (1, num_classes)

    # Ensure parameters is 2D and has the correct shape (1, 6)
    if parameters.dim() == 1:
        parameters = parameters.unsqueeze(0)  # Shape: (1, param_size)

    # Ensure parameters has the correct shape
    if parameters.shape[0] != 1:
        parameters = parameters[:1, :]  # Take only the first row if more than one exists
    elif parameters.shape[0] == 1 and parameters.shape[1] == 6:
        parameters = parameters
    else:
        parameters = parameters.view(1, 6)  # Ensure parameters has the shape [1, 6]

    embedding_vector = torch.cat((one_hot_tensor, parameters), dim=1).cuda()
    assert embedding_vector.shape == (1, 11), f"Expected shape-> (1, 11), got {embedding_vector.shape}"
    
    return embedding_vector


class CommandTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6):
        super(CommandTransformer, self).__init__()
        self.point_cloud_encoder = PointCloudTransformerLayer().cuda()
        self.pos_encoder = PositionalEncoding(d_model).cuda()
        self.transformer = CustomTransformerDecoder(d_model, d_model, d_model, num_layers, 2048).cuda()
        self.output_layer = TransformerOutputLayer(d_model).cuda()
        self.linear = nn.Linear(11, 512).cuda()  # Linear layer to project each element of tgt to d_model dimensions

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        src_emb = src.unsqueeze(0).cuda()  # Adding batch dimension
        # print(f"Original tgt shape: {tgt.shape}")

        tgt = tgt.view(-1, 11)  # Shape: (batch_size*sequence_length, 11)
        # tgt = self.linear(tgt)  # Shape: (batch_size*sequence_length, 512)
        tgt = tgt.view(1, -1, 512)  # Reshape back to (1, N*11, 512)
        # print(f"Shape after linear layer: {tgt.shape}")

        tgt_emb = self.pos_encoder(tgt)
        # print(f"Shape after positional encoding: {tgt_emb.shape}")

        # Transformer forward pass
        transformer_output = self.transformer(tgt_emb, src_emb)
        # print(f"Transformer input shape: {tgt_emb.shape}")
        # print(f"Transformer output shape: {transformer_output.shape}")

        # Output layer forward pass
        outputs = self.output_layer(transformer_output)
        # print(f"Output layer shapes: command_probs: {outputs[0].shape}, parameters_probs: {outputs[1].shape}")

        return outputs


if __name__ == "__main__":

    model = CommandTransformer(vocab_size=VOCAB_SIZE).cuda()
    input_emb = construct_embedding_vector_from_vocab(Commands.START, torch.zeros(6).cuda()).cuda()
    input_emb_proj = model.linear(input_emb).unsqueeze(1)  # Shape: (1, 1, 512)
    input_emb_proj = input_emb_proj.repeat(1, 11, 1)  # Shape: (1, 11, 512)

    while True:
        pred = model(pt_cloud_encoded_features, input_emb_proj)
        command, parameters = select_parameters(*pred)
        print(command)
        print(parameters[0])
        output_emb = construct_embedding_vector_from_vocab(command, parameters).cuda()

        # Apply linear layer to project the new embedding to 512 dimensions
        output_emb_proj = model.linear(output_emb).unsqueeze(1)  # Shape: (1, 1, 512)
        output_emb_proj = output_emb_proj.repeat(1, 11, 1)  # Shape: (1, 11, 512)

        # Concatenate along the sequence dimension
        input_emb_proj = torch.cat((input_emb_proj, output_emb_proj), dim=1).cuda()

        if command == Commands.STOP:
            break

    # print(input_emb_proj)
    print(input_emb_proj.shape)