import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from enum import Enum
from three_d_scene_script.pt_cloud_encoder import PointCloudTransformerLayer
from test_gt_N import SceneScriptProcessor  
import plotly.graph_objects as go

# Ensure all operations are done on the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the point cloud encoder model
encoder_model = PointCloudTransformerLayer().to(device)
pt_cloud_path = "/home/mseleem/Desktop/3d_model_pt/0/semidense_points.csv.gz"
points, dist_std = encoder_model.read_points_file(pt_cloud_path)
sparse_tensor = encoder_model.process_point_cloud(points, dist_std)
pt_cloud_encoded_features = encoder_model(sparse_tensor).to(device)

# Define Commands enum
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

    @classmethod
    def get_one_hot(cls, command_type: str):
        command = cls[command_type.upper()]
        one_hot_vector = torch.zeros(len(cls)).to(device)
        one_hot_vector[command.value - 1] = 1
        return one_hot_vector

# Positional encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# Cross-attention mechanism
class CrossAttention(nn.Module):
    def __init__(self, d_model, d_out_kq, d_out_v):
        super(CrossAttention, self).__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Linear(d_model, d_out_kq).to(device)
        self.W_key = nn.Linear(d_model, d_out_kq).to(device)
        self.W_value = nn.Linear(d_model, d_out_v).to(device)

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

# Transformer decoder layer with self-attention and cross-attention
class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, d_out_kq, d_out_v, dim_feedforward=2048):
        super(CustomTransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True).to(device)
        self.cross_attention = CrossAttention(d_model, d_out_kq, d_out_v).to(device)
        self.linear1 = nn.Linear(d_model, dim_feedforward).to(device)
        self.linear2 = nn.Linear(dim_feedforward, d_model).to(device)
        self.norm1 = nn.LayerNorm(d_model).to(device)
        self.norm2 = nn.LayerNorm(d_model).to(device)
        self.norm3 = nn.LayerNorm(d_model).to(device)

    def forward(self, tgt, memory, tgt_mask=None):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + tgt2)
        tgt2 = self.cross_attention(tgt, memory)
        tgt = self.norm2(tgt + tgt2)
        tgt2 = self.linear2(F.relu(self.linear1(tgt)))
        tgt = self.norm3(tgt + tgt2)
        return tgt

# Transformer decoder
class CustomTransformerDecoder(nn.Module):
    def __init__(self, d_model, d_out_kq, d_out_v, num_decoder_layers, dim_feedforward):
        super(CustomTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            CustomTransformerDecoderLayer(d_model, d_out_kq, d_out_v, dim_feedforward) for _ in range(num_decoder_layers)
        ]).to(device)

    def forward(self, tgt, memory, tgt_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask)
        return tgt

class CommandTransformer(nn.Module):
    def __init__(self, d_model=512, num_layers=6):
        super(CommandTransformer, self).__init__()
        self.point_cloud_encoder = PointCloudTransformerLayer().to(device)
        self.pos_encoder = PositionalEncoding(d_model).to(device)

        # The input dimension is determined dynamically
        self.input_dim = None  # This will be set later based on the actual input
        self.d_model = d_model

        self.transformer = CustomTransformerDecoder(d_model, d_model, d_model, num_layers, 2048).to(device)
        self.final_linear = None  # To be set dynamically

    def set_input_dim(self, input_dim):
        """Sets the input dimension dynamically and initializes related layers."""
        self.input_dim = input_dim
        self.initial_linear = nn.Linear(self.input_dim, self.d_model).to(device)  # Project input to d_model
        self.final_linear = nn.Linear(self.d_model, self.input_dim).to(device)  # Project back to input_dim

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, tgt_mask=None):
        if self.input_dim is None:
            raise ValueError("Input dimension is not set. Call `set_input_dim` with the correct input dimension.")

        src_emb = src.to(device)
        tgt_emb = self.initial_linear(tgt).to(device)  # Project input to d_model

        # tgt_emb should now have shape [batch_size, sequence_length, d_model]
        tgt_emb = self.pos_encoder(tgt_emb)
        transformer_output = self.transformer(tgt_emb, src_emb, tgt_mask=tgt_mask)
        final_output = self.final_linear(transformer_output)  # Project back to [batch_size, sequence_length, input_dim]

        return final_output

# Generate subsequent mask for the transformer
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).to(device)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

processor = SceneScriptProcessor('/home/mseleem/Desktop/3d_SceneScript/0/ase_scene_language.txt')
decoder_input_embeddings, gt_output_embeddings = processor.process()
decoder_input_embeddings = decoder_input_embeddings.to(device)
gt_output_embeddings = gt_output_embeddings.to(device)
model = CommandTransformer(d_model=512, num_layers=6).to(device)
input_dim = decoder_input_embeddings.size(-1)  
model.set_input_dim(input_dim)

command_dim = len(Commands)  
parameter_dim = input_dim - command_dim  
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
criterion_command = nn.CrossEntropyLoss()  
criterion_parameters = nn.MSELoss() 

num_epochs = 200
total_loss_list = []
command_loss_list = []
parameter_loss_list = []

last_epoch_predictions = []
last_epoch_ground_truths = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    total_command_loss = 0.0
    total_parameter_loss = 0.0
    for t in range(decoder_input_embeddings.size(1)):
        current_input = decoder_input_embeddings[:, t:t+1, :].to(device)  
        tgt_mask = generate_square_subsequent_mask(1) 
        output = model(
            src=pt_cloud_encoded_features, tgt=current_input, tgt_mask=tgt_mask)  

        gt_output = gt_output_embeddings[:, t:t+1, :].to(device) 
        output_command = output[:, :, :command_dim] 
        output_parameters = output[:, :, command_dim:]  
        gt_command = gt_output[:, :, :command_dim].argmax(dim=-1) 
        gt_parameters = gt_output[:, :, command_dim:]  
        loss_command = criterion_command(output_command.view(-1, command_dim), gt_command.view(-1))
        loss_parameters = criterion_parameters(output_parameters.view(-1, parameter_dim), gt_parameters.view(-1, parameter_dim))
        loss = loss_command + loss_parameters
        loss.backward()
        total_loss += loss.item()
        total_command_loss += loss_command.item()
        total_parameter_loss += loss_parameters.item()

        # Store predictions and ground truths for the last epoch
        if epoch == num_epochs - 1:
            last_epoch_predictions.append((output_command.argmax(dim=-1), output_parameters))
            last_epoch_ground_truths.append((gt_command, gt_parameters))

    optimizer.step()
    scheduler.step()
    total_loss_list.append(total_loss)
    command_loss_list.append(total_command_loss)
    parameter_loss_list.append(total_parameter_loss)
    print(f'Epoch {epoch + 1}/{num_epochs}, Total Loss: {total_loss}, Command Loss: {total_command_loss}, '
          f'Parameter Loss: {total_parameter_loss}')

# Print predictions and ground truths for the last epoch
print("\nPredictions and Ground Truths for the last epoch:")
for timestep, (pred, gt) in enumerate(zip(last_epoch_predictions, last_epoch_ground_truths)):
    pred_command, pred_parameters = pred
    gt_command, gt_parameters = gt
    print(f"Timestep {timestep + 1}:")
    print(f"Predicted Command: {pred_command.detach().cpu().numpy()}, Predicted Parameters: {pred_parameters.detach().cpu().numpy()}")
    print(f"Ground Truth Command: {gt_command.detach().cpu().numpy()}, Ground Truth Parameters: {gt_parameters.detach().cpu().numpy()}")


fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(num_epochs)), y=total_loss_list, mode='lines', name='Total Loss', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=list(range(num_epochs)), y=command_loss_list, mode='lines', name='Command Loss', line=dict(color='green')))
fig.add_trace(go.Scatter(x=list(range(num_epochs)), y=parameter_loss_list, mode='lines', name='Parameter Loss', line=dict(color='red')))
fig.update_layout(
    title='Training Loss over Epochs',
    xaxis_title='Epoch',
    yaxis_title='Loss',
    hovermode='x',
)
fig.show()


