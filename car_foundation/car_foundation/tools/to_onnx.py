import torch
import torch.onnx
import torch.nn as nn

from car_foundation.models import TorchTransformerDecoder, TorchGPT2

model_path = "MODEL-PATH"
state_dim = 6
action_dim = 2
latent_dim = 128
num_heads = 4
num_layers = 3
dropout = 0.1
history_length = 250
prediction_length = 50
batch_size = 200
device = torch.device("cpu")

model = TorchTransformerDecoder(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, device, dropout, history_length, prediction_length)
model.load_state_dict(torch.load(model_path))
model.eval()

history_input = torch.randn(batch_size, history_length, state_dim + action_dim)
history_mask = torch.ones(batch_size, history_length * 2 - 1)
prediction_input = torch.randn(batch_size, prediction_length, action_dim)
prediction_mask = torch.ones(batch_size, prediction_length)
tgt_mask = nn.Transformer.generate_square_subsequent_mask(prediction_length)

torch.onnx.export(model, (history_input, prediction_input, history_mask, prediction_mask, tgt_mask), "model.onnx", verbose=True)
