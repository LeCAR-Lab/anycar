from jaxonnxruntime import backend as jax_backend
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import time
import onnx
import matplotlib.pyplot as plt

from torch2jax import j2t, t2j

from car_foundation.models import TorchTransformerDecoder, TorchTransformerDecoder2
from car_foundation.jax_models import JaxTransformerDecoder

print(jax.devices())

model_path = "MODEL-PATH"
onnx_model_path = "MODEL-PATH.onnx"
state_dim = 6
action_dim = 2
latent_dim = 64
num_heads = 4
num_layers = 2
dropout = 0.1
history_length = 250
prediction_length = 50
batch_size = 1500

rng = jax.random.PRNGKey(0)
rng, params_rng = jax.random.split(rng)
rng, dropout_rng = jax.random.split(rng)
init_rngs = {'params': params_rng, 'dropout': dropout_rng}

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch_model = TorchTransformerDecoder2(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, torch_device, dropout, history_length, prediction_length)
# torch_model.load_state_dict(torch.load(model_path))
# torch_model.eval()
# torch_model.to(torch_device)

torch_history_input = torch.randn(batch_size, history_length, state_dim + action_dim, requires_grad=False).to(torch_device)
torch_history_mask = torch.ones(batch_size, history_length * 2 - 1, requires_grad=False).to(torch_device)
torch_prediction_input = torch.randn(batch_size, prediction_length, action_dim, requires_grad=False).to(torch_device)
torch_prediction_mask = torch.ones(batch_size, prediction_length, requires_grad=False).to(torch_device)
torch_tgt_mask = nn.Transformer.generate_square_subsequent_mask(prediction_length).to(torch_device).requires_grad_(False)

jax_history_input = jnp.array(torch_history_input.cpu().numpy(), dtype=jnp.bfloat16)
jax_history_mask = jnp.array(torch_history_mask.cpu().numpy(), dtype=jnp.bfloat16)
jax_prediction_input = jnp.array(torch_prediction_input.cpu().numpy(), dtype=jnp.bfloat16)
jax_prediction_mask = jnp.array(torch_prediction_mask.cpu().numpy(), dtype=jnp.bfloat16)
jax_tgt_mask = jnp.array(torch_tgt_mask.cpu().numpy(), dtype=jnp.bfloat16)
jax_model = JaxTransformerDecoder(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, dropout, history_length, prediction_length)
variables = jax_model.init(init_rngs, jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask, jax_tgt_mask)

ave_time = 0
max_time = float("-inf")
min_time = float("inf")

jit_func = jax.jit(jax_model.apply)

for i in range(5):
    with torch.no_grad():
        # jax_model.apply(variables, jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask, jax_tgt_mask, rngs=init_rngs)
        jit_func(variables, jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask, jax_tgt_mask, rngs=init_rngs)
times = []
for i in range(100):
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()
        # jax_model.apply(variables, jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask, jax_tgt_mask, rngs=init_rngs)
        jit_func(variables, jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask, jax_tgt_mask, rngs=init_rngs)
        torch.cuda.synchronize()
        end = time.time()
    time.sleep(0.02)
    time_diff = end - start
    ave_time += time_diff
    max_time = max(max_time, time_diff)
    min_time = min(min_time, time_diff)
    times.append(time_diff)
ave_time /= 100
plt.figure()
plt.plot(times)
plt.savefig('inference_times.png')
print(f"PyTorch average inference time: {ave_time * 1000} ms")
print(f"PyTorch max inference time: {max_time * 1000} ms")
print(f"PyTorch min inference time: {min_time * 1000} ms")

# onnx_model = onnx.load(onnx_model_path)
# jax_model = jax_backend.prepare(onnx_model, device="gpu")

# jax_history_input = jnp.array(torch_history_input.cpu().numpy())
# jax_history_mask = jnp.array(torch_history_mask.cpu().numpy())
# jax_prediction_input = jnp.array(torch_prediction_input.cpu().numpy())
# jax_prediction_mask = jnp.array(torch_prediction_mask.cpu().numpy())
# jax_tgt_mask = jnp.array(torch_tgt_mask.cpu().numpy())

# put tensor on gpu


# ave_time = 0
# max_time = float("-inf")
# min_time = float("inf")

# jax_model.run((jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask, jax_tgt_mask))

# @jax.jit
# def run_model(jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask, jax_tgt_mask):
#     return jax_model.run((jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask, jax_tgt_mask))

# for i in range(100):
#     start = time.time()
#     run_model(jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask, jax_tgt_mask)
#     end = time.time()
#     time_diff = end - start
#     ave_time += time_diff
#     max_time = max(max_time, time_diff)
#     min_time = min(min_time, time_diff)
# ave_time /= 100

# print(f"JAX average inference time: {ave_time * 1000} ms")
# print(f"JAX max inference time: {max_time * 1000} ms")
# print(f"JAX min inference time: {min_time * 1000} ms")
