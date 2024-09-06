import torch
import torch.nn as nn
from torch.nn import functional as F
from three_d_scene_script.decoder_module import CommandTransformer, generate_square_subsequent_mask

model = CommandTransformer()

input_dim = 512
model.set_input_dim(input_dim)

print("Model Summary:")
print(model)

batch_size = 2
seq_len = 5
src = torch.rand(batch_size, seq_len, input_dim).to(model.fc1.weight.device)
tgt = torch.rand(batch_size, seq_len, input_dim).to(model.fc1.weight.device)
tgt_mask = generate_square_subsequent_mask(seq_len)
with torch.no_grad():
    final_output = model(src, tgt, tgt_mask=tgt_mask)
print("\nFinal Output:")
print(final_output)
print("\nWeights of the first fully connected layer (fc1):")
print(model.fc1.weight)
# print("\nWeights of the second fully connected layer (fc2):")
# print(model.fc2.weight)
# print("\nWeights of the third fully connected layer (fc3):")
# print(model.fc3.weight)
print("\nWeights of the final linear layer:")
print(model.final_linear.weight)
x = F.relu(model.fc1(final_output))
print("\nOutput after third fully connected layer (fc3):")
print(x)
final_output = model.final_linear(x)
print("\nFinal output after passing through all layers:")
print(final_output)


