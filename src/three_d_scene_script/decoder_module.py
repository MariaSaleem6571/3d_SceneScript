import os
import torch
from three_d_scene_script.pt_cloud_encoder import PointCloudTransformerLayer
import torch.nn as nn
import torch.nn.functional as F
import math
from enum import Enum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Commands(Enum):
    START = 1
    STOP = 2
    MAKE_WALL = 3
    MAKE_WINDOW = 4
    MAKE_DOOR = 5

    @classmethod
    def get_name_for(cls, value: int):
        """
        Get the name of the command for the given value

        :param value: The value of the command
        :return: The name of the command
        """

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
        """
        Initialize the output layer

        :param transformer_dim: The dimension of the transformer
        """

        super(TransformerOutputLayer, self).__init__()
        self.command_layer = nn.Linear(transformer_dim, 5)
        self.param_layer = nn.Linear(transformer_dim, 6)

    def forward(self, x):
        """
        Forward pass

        :param x: The input tensor
        :return: command_probs, parameters_probs
        """

        command_logits = self.command_layer(x)
        command_logits[..., 0] = -float('inf')

        #command_probs = F.softmax(command_logits, dim=-1)
        command_probs = command_logits
        parameter_logits = self.param_layer(x)
        parameters_probs = torch.tanh(parameter_logits)
        #parameters_probs = parameter_logits
        return command_probs, parameters_probs

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Initialize the positional encoding

        :param d_model: The model dimension
        :param max_len: The maximum length
        """

        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass

        :param x: The input tensor
        :return: x + self.pe[:x.size(0), :]
        """

        x = x + self.pe[:x.size(0), :]
        return x

class CrossAttention(nn.Module):

    def __init__(self, d_model, d_out_kq, d_out_v):
        """
        Initialize the cross attention

        :param d_model: The model dimension
        :param d_out_kq: The output dimension for key and query
        :param d_out_v: The output dimension for value
        """

        super(CrossAttention, self).__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Linear(d_model, d_out_kq).to(device)
        self.W_key = nn.Linear(d_model, d_out_kq).to(device)
        self.W_value = nn.Linear(d_model, d_out_v).to(device)

    def forward(self, x_1, x_2, attn_mask=None):
        """
        Forward pass

        :param x_1: The first input tensor
        :param x_2: The second input tensor
        :param attn_mask: The attention mask
        :return: context_vec
        """

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
        """
        Initialize the custom transformer decoder layer

        :param d_model: The model dimension
        :param d_out_kq: The output dimension for key and query
        :param d_out_v: The output dimension for value
        :param dim_feedforward: The feedforward dimension
        """

        super(CustomTransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True).to(device)
        self.cross_attention = CrossAttention(d_model, d_out_kq, d_out_v).to(device)
        self.linear1 = nn.Linear(d_model, dim_feedforward).to(device)
        self.linear2 = nn.Linear(dim_feedforward, d_model).to(device)
        self.norm1 = nn.LayerNorm(d_model).to(device)
        self.norm2 = nn.LayerNorm(d_model).to(device)
        self.norm3 = nn.LayerNorm(d_model).to(device)

    def forward(self, tgt, memory, tgt_mask=None):
        """
        Forward pass

        :param tgt: The target tensor
        :param memory: The memory tensor
        :param tgt_mask: The target mask
        :return: tgt
        """

        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + tgt2)
        tgt2 = self.cross_attention(tgt, memory)
        tgt = self.norm2(tgt + tgt2)
        tgt2 = self.linear2(F.relu(self.linear1(tgt)))
        tgt = self.norm3(tgt + tgt2)
        return tgt

class CustomTransformerDecoder(nn.Module):
    def __init__(self, d_model, d_out_kq, d_out_v, num_decoder_layers, dim_feedforward):
        """
        Initialize the custom transformer decoder
        :param d_model: The model dimension
        :param d_out_kq: The output dimension for key and query
        :param d_out_v: The output dimension for value
        :param num_decoder_layers: The number of decoder layers
        :param dim_feedforward: The feedforward dimension
        """

        super(CustomTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            CustomTransformerDecoderLayer(d_model, d_out_kq, d_out_v, dim_feedforward) for _ in range(num_decoder_layers)]).to(device)

    def forward(self, tgt, memory, tgt_mask=None):
        """
        Forward pass

        :param tgt: The target tensor
        :param memory: The memory tensor
        :param tgt_mask: The target mask
        :return: tgt
        """

        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask)
        return tgt

def select_parameters(command_probs, parameters_probs):
    """
    Select the parameters

    :param command_probs: The command probabilities
    :param parameters_probs: The parameters probabilities
    :return: command_name, parameters
    """

    command_indx = command_probs[0, 0].argmax(dim=-1).item() + 1
    if command_indx == Commands.STOP.value:
        parameters = torch.zeros(6).cuda()
    elif command_indx in [Commands.MAKE_WALL.value, Commands.MAKE_DOOR.value, Commands.MAKE_WINDOW.value]:
        parameters = parameters_probs.squeeze()
    else:
        parameters = torch.zeros(6).cuda()
    return Commands.get_name_for(command_indx), parameters

class CommandTransformer(nn.Module):

    def __init__(self, d_model=512, num_layers=6):
        """
        Initialize the command transformer

        :param d_model: The model dimension
        :param num_layers: The number of layers
        """
        super(CommandTransformer, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model).to(device)
        self.d_model = d_model
        self.transformer = CustomTransformerDecoder(d_model, d_model, d_model, num_layers, 2048).to(device)
        self.fc1 = nn.Linear(d_model, d_model).to(device)
        self.initial_linear = None
        self.final_linear = None

    def set_input_dim(self, input_dim):
        """
        Set the input dimension and initialize required layers

        :param input_dim: The input dimension
        """
        self.initial_linear = nn.Linear(input_dim, self.d_model).to(device)
        self.output_layer = TransformerOutputLayer(self.d_model).to(device)
        self.final_linear = nn.Linear(self.d_model, input_dim).to(device)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, tgt_mask=None):
        """
        Forward pass

        :param src: The source tensor (encoded point cloud)
        :param tgt: The target tensor (input embeddings)
        :param tgt_mask: The target mask for transformer
        :return: final_output
        """
        src_emb = src.to(device)
        tgt_emb = self.initial_linear(tgt).to(device)
        tgt_emb = self.pos_encoder(tgt_emb)
        transformer_output = self.transformer(tgt_emb, src_emb, tgt_mask=tgt_mask)

        x = F.relu(self.fc1(transformer_output))
        final_output = self.final_linear(transformer_output)

        return final_output

def generate_square_subsequent_mask(sz):
    """
    Generate a square subsequent mask

    :param sz: The size
    :return: mask
    """

    mask = torch.triu(torch.ones(sz, sz), diagonal=1).to(device)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

def construct_embedding_vector_from_vocab(command: Commands, parameters: torch.Tensor):
    """
    Construct the embedding vector from the vocabulary

    :param command: The command
    :param parameters: The parameters
    :return: embedding_vector
    """

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
