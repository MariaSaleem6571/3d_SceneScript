import os
import torch
from three_d_scene_script.pt_cloud_encoder import PointCloudTransformerLayer
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
        self.param_layer = nn.Linear(transformer_dim, 6) 

    def forward(self, x):
        command_logits = self.command_layer(x)  
        command_logits[..., 0] = -float('inf')

        command_probs = F.softmax(command_logits, dim=-1)  
        parameter_logits = self.param_layer(x)  
        parameters_probs = torch.tanh(parameter_logits)  
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

    def forward(self, x_1, x_2, attn_mask=None):
        queries_1 = self.W_query(x_1)
        keys_2 = self.W_key(x_2)
        values_2 = self.W_value(x_2)
        attn_scores = queries_1.matmul(keys_2.transpose(-2, -1))
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == float('-inf'), float('-inf'))
        attn_weights = torch.softmax(attn_scores / self.d_out_kq ** 0.5, dim=-1)
        context_vec = attn_weights.matmul(values_2)
        return context_vec


class CustomTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, d_out_kq, d_out_v, dim_feedforward=2048):
        super(CustomTransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        self.cross_attention = CrossAttention(d_model, d_out_kq, d_out_v)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt)
        tgt2 = self.cross_attention(tgt, memory) 
        tgt = self.norm2(tgt)
        tgt2 = self.linear2((F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt)
        return tgt

class CustomTransformerDecoder(nn.Module):

    def __init__(self, d_model, d_out_kq, d_out_v, num_decoder_layers, dim_feedforward):
        super(CustomTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            CustomTransformerDecoderLayer(d_model, d_out_kq, d_out_v, dim_feedforward) for _ in
            range(num_decoder_layers)])

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return tgt


def construct_embedding_vector_from_vocab(command: Commands, parameters: torch.Tensor):
    num_classes = len(Commands)
    value_tensor = torch.tensor(command.value - 1).cuda()
    one_hot_tensor = F.one_hot(value_tensor, num_classes=num_classes).float().cuda()
    if one_hot_tensor.dim() == 1:
        one_hot_tensor = one_hot_tensor.unsqueeze(0) 
    if parameters.dim() == 1:
        parameters = parameters.unsqueeze(0) 
    if parameters.shape[0] != 1:
        parameters = parameters[:1, :]
    elif parameters.shape[0] == 1 and parameters.shape[1] == 6:
        parameters = parameters
    else:
        parameters = parameters.view(1, 6)  
    embedding_vector = torch.cat((one_hot_tensor, parameters), dim=1).cuda()
    return embedding_vector

def generate_square_subsequent_mask(sz):
    """Generate a square mask for the sequence. Mask out subsequent positions."""
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.cuda()

class CommandTransformer(nn.Module):

    def __init__(self, vocab_size, d_model=512, num_layers=6):
        super(CommandTransformer, self).__init__()
        self.point_cloud_encoder = PointCloudTransformerLayer().cuda()
        self.pos_encoder = PositionalEncoding(d_model).cuda()
        self.transformer = CustomTransformerDecoder(d_model, d_model, d_model, num_layers, 2048).cuda()
        self.initial_linear = nn.Linear(vocab_size, d_model).cuda()  
        self.output_layer = TransformerOutputLayer(d_model).cuda()
        self.final_linear = nn.Linear(d_model, vocab_size).cuda() 

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, tgt_mask=None):
        src_emb = src.cuda() 
        tgt_emb = self.initial_linear(tgt).cuda() 
        tgt_emb = self.pos_encoder(tgt_emb)
        transformer_output = self.transformer(tgt_emb, src_emb, tgt_mask=tgt_mask)  
        command_probs, parameters_probs = self.output_layer(transformer_output)
        final_output = self.final_linear(transformer_output) 
        return command_probs, parameters_probs, final_output

model = CommandTransformer(vocab_size=VOCAB_SIZE).cuda()
input_emb = construct_embedding_vector_from_vocab(Commands.START, torch.zeros(1,6).cuda()).cuda()
input_emb_proj = model.initial_linear(input_emb).unsqueeze(0)  
final_output = model.final_linear(input_emb_proj) 
accumulated_output = final_output  
max_iterations = 150

for i in range(max_iterations):
    tgt_mask = generate_square_subsequent_mask(accumulated_output.size(1)) 
    command_probs, parameters_probs, final_output = model(pt_cloud_encoded_features, accumulated_output, tgt_mask=tgt_mask)
    command, parameters = select_parameters(command_probs, parameters_probs)
    print(f"Selected command: {command}, parameters: {parameters[0]}")
    if command == Commands.STOP:
        accumulated_output = torch.cat((accumulated_output, final_output[:, -1:, :]), dim=1)
        break
    accumulated_output = torch.cat((accumulated_output, final_output[:, -1:, :]), dim=1)
    if i + 1 >= max_iterations:
        print("Reached maximum iterations")
        break
print(f"All predictions shape: {accumulated_output.shape}")  
