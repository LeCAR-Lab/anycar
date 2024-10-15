import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax import grad, jit, vmap
from functools import partial
from car_planner.fast_spline_trajectory_generation import interpolate_action_sequence
from car_foundation.utils import generate_subsequences_hf
from transformers import GPT2Config, GPT2Model, FlaxGPT2Model, modeling_flax_pytorch_utils
from transformers.models.gpt2.modeling_flax_gpt2 import FlaxGPT2Module
from flax import linen

from typing import Sequence, Optional
import os
import time
import math
import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn, Tensor

from collections import OrderedDict

class MLP(linen.Module):
    features: Sequence[int]

    @linen.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = linen.relu(linen.Dense(feat)(x))
        return linen.Dense(self.features[-1])(x)
    
class TorchMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)
    
class TorchTransformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, action_dim, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        self.input_dim = output_dim + action_dim
        self.embedding = nn.Linear(self.input_dim, embed_dim)
        
        self.pos_encoder = nn.Parameter(torch.zeros(input_dim, embed_dim))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)  # x shape: [batch_size, seq_length, input_dim]
        x += self.pos_encoder[:x.size(1)]  # add positional encoding
        x = self.transformer_encoder(x)  # pass through transformer
        x = self.output_layer(x[:, -1, :])  # take the last timestep output and predict next state
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, flip=False):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(max_len, d_model))
        self.dropout = nn.Dropout(0.1)
        self.flip = flip

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        if self.flip:
            x = x + torch.flip(self.pe[:x.size(1)], [0])
        else:
            x = x + self.pe[:x.size(1)]
        return self.dropout(x)
    
class TorchTransformerDecoder(nn.Module):
    def __init__(self, state_dim, action_dim, output_dim, latent_dim, num_heads, num_layers, device, dropout=0.1, history_length=250, prediction_length=50):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.history_length = history_length
        self.prediction_length = prediction_length

        self.odd_indices = torch.arange(0, history_length * 2 - 1, 2, device=device)
        self.even_indices = torch.arange(1, history_length * 2 - 1, 2, device=device)

        self.state_embedding = nn.Linear(state_dim, latent_dim)
        self.action_embedding = nn.Linear(action_dim, latent_dim)
        self.output_embedding = nn.Linear(latent_dim, self.output_dim)
        
        self.history_pos_emb = LearnedPositionalEncoding(latent_dim, history_length * 2 - 1, flip=True)
        self.action_pos_emb = LearnedPositionalEncoding(latent_dim, prediction_length)
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=num_layers)

    def forward(self, history, action, history_padding_mask=None, action_padding_mask=None, tgt_mask=None):
        # history_emb = torch.zeros(history.size(0), history.size(1) * 2 - 1, self.latent_dim, device=history.device)
        # history_emb[self.odd_indices] = self.state_embedding(history[:, :, :self.state_dim]) # shape: [batch_size, seq_length, latent_dim]
        # history_emb[self.even_indices] = self.action_embedding(history[:, :-1, self.state_dim:self.state_dim+self.action_dim]) # shape: [batch_size, seq_length-1, latent_dim]
        state_emb = self.state_embedding(history[:, :, :self.state_dim])
        action_emb = self.action_embedding(history[:, :, self.state_dim:self.state_dim+self.action_dim])
        history_emb = torch.cat((state_emb[:, None, :, :], action_emb[:, None, :, :]), dim=1).view(-1, 2 * self.history_length, self.latent_dim).transpose(1, 2).contiguous().view(-1, 2 * self.history_length, self.latent_dim)[:, :-1, :]
        
        history_emb = self.history_pos_emb(history_emb)

        action_emb = self.action_embedding(action)
        action_emb = self.action_pos_emb(action_emb)

        x = self.transformer_decoder(action_emb, history_emb,
                                     tgt_is_causal=True, # memory_is_causal=True,
                                     tgt_mask = tgt_mask,
                                     # memory_mask = nn.Transformer.generate_square_subsequent_mask(history_emb.size(1), device=history_emb.device),
                                     tgt_key_padding_mask=action_padding_mask,
                                     memory_key_padding_mask=history_padding_mask
                                    )
        x = self.output_embedding(x)
        return x
                                     
class TorchTransformer(nn.Module):
    def __init__(self, history_dim, action_dim, output_dim, latent_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.history_dim = history_dim
        self.action_dim = action_dim
        self.output_dim = output_dim

        self.history_embedding = nn.Linear(history_dim, latent_dim)
        self.action_embedding = nn.Linear(action_dim, latent_dim)
        self.output_embedding = nn.Linear(latent_dim, self.output_dim)
        
        self.pos_emb = PositionalEncoding(latent_dim)
        self.transformer = nn.Transformer(
            d_model=latent_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=128, dropout=dropout, batch_first=True
        )

    def forward(self, history, action, history_padding_mask=None, action_padding_mask=None):
        history_emb = self.history_embedding(history)
        history_emb = self.pos_emb(history_emb)
        action_emb = self.action_embedding(action)
        action_emb = self.pos_emb(action_emb)

        x = self.transformer(history_emb, action_emb,
                             src_is_causal=True, tgt_is_causal=True, memory_is_causal=True,
                             src_mask = nn.Transformer.generate_square_subsequent_mask(history_emb.size(1), device=history_emb.device),
                             tgt_mask = nn.Transformer.generate_square_subsequent_mask(action_emb.size(1), device=action_emb.device),
                             # memory_mask = nn.Transformer.generate_square_subsequent_mask(history_emb.size(1), device=history_emb.device),
                             src_key_padding_mask=history_padding_mask,
                             tgt_key_padding_mask=action_padding_mask,
                             memory_key_padding_mask=history_padding_mask)
        x = self.output_embedding(x)
        return x
    

class TorchGPT2(GPT2Model):
    def __init__(self, state_dim, action_dim, output_dim, latent_dim, num_heads, num_layers, dropout=0.1):
        config = GPT2Config(
            vocab_size=state_dim + action_dim,
            n_positions=550,
            n_embd=latent_dim,
            n_layer=num_layers,
            n_head=num_heads,
            n_inner=4 * latent_dim,
            activation_function='gelu_new',
            resid_pdrop = dropout,
            embd_pdrop = dropout,
            attn_pdrop = dropout,
            use_cache = False, # setting to true will be useful later for inference
        )
        super().__init__(config)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim

        self.state_embedding = nn.Linear(state_dim, latent_dim)
        self.action_embedding = nn.Linear(action_dim, latent_dim)
        self.output_embedding = nn.Linear(latent_dim, self.output_dim)
        self.seperator_token = nn.Parameter(torch.randn(1, 1, latent_dim))

    def forward(self, history, action, history_padding_mask=None, action_padding_mask=None):
        history_emb = torch.zeros(history.size(0), history.size(1) * 2 - 1, self.latent_dim, device=history.device)
        history_emb[:, ::2] = self.state_embedding(history[:, :, :self.state_dim]) # shape: [batch_size, seq_length, latent_dim]
        history_emb[:, 1::2] = self.action_embedding(history[:, :-1, self.state_dim:self.state_dim+self.action_dim]) # shape: [batch_size, seq_length-1, latent_dim]
        action_emb = self.action_embedding(action)
        seperator_token = self.seperator_token.expand(action_emb.size(0), 1, self.latent_dim)
        inputs_embeds = torch.cat([history_emb, seperator_token, action_emb], dim=1)
        if history_padding_mask is None:
            history_padding_mask = torch.ones(history_emb.size(0), history_emb.size(1), dtype=torch.float32, device=history_emb.device)
        if action_padding_mask is None:
            action_padding_mask = torch.ones(action_emb.size(0), action_emb.size(1), dtype=torch.float32, device=action_emb.device)
        seperator_token_mask = torch.ones(action_emb.size(0), 1, dtype=torch.float32, device=action_emb.device)
        attn_mask = torch.cat([history_padding_mask, seperator_token_mask, action_padding_mask], dim=1)

        x = super().forward(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
        x = self.output_embedding(x.last_hidden_state[:, -action.size(1):, :])
        return x
    
class CustomFlaxGPT2Module(FlaxGPT2Module):
    state_dim: int = 6
    action_dim: int = 2
    output_dim: int = 6
    latent_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    dropout_percent: float = 0.1

    def setup(self):
        FlaxGPT2Module.setup(self)
        self.state_embedding = linen.Dense(self.latent_dim, name='state_embedding')
        self.action_embedding = linen.Dense(self.latent_dim, name='action_embedding')
        self.output_embedding = linen.Dense(self.output_dim, name='output_embedding')
        self.seperator_token = self.param('seperator_token', linen.initializers.xavier_uniform(), (1, 1, self.latent_dim))

    def forward(self, history, action, history_padding_mask=None, action_padding_mask=None, **kwargs):
        history_emb = jnp.zeros((history.shape[0], history.shape[1] * 2 - 1, self.latent_dim))
        history_emb = history_emb.at[:, ::2].set(self.state_embedding(history[:, :, :self.state_dim])) # shape: [batch_size, seq_length, latent_dim]
        history_emb = history_emb.at[:, 1::2].set(self.action_embedding(history[:, :-1, self.state_dim:self.state_dim+self.action_dim])) # shape: [batch_size, seq_length-1, latent_dim]
        action_emb = self.action_embedding(action)
        seperator_token = jnp.repeat(self.seperator_token, action_emb.shape[0], axis=0)
        inputs_embeds = jnp.concatenate([history_emb, seperator_token, action_emb], axis=1)
        if history_padding_mask is None:
            history_padding_mask = jnp.ones((history_emb.shape[0], history_emb.shape[1]))
        if action_padding_mask is None:
            action_padding_mask = jnp.ones((action_emb.shape[0], action_emb.shape[1]))
        seperator_token_mask = jnp.ones((action_emb.shape[0], 1))
        attn_mask = jnp.concatenate([history_padding_mask, seperator_token_mask, action_padding_mask], axis=1)

        x = FlaxGPT2Module.__call__(self, inputs_embeds=inputs_embeds, attention_mask=attn_mask, **kwargs)
        x = self.output_embedding(x.last_hidden_state[:, -action.shape[1]:, :])
        return x
    
    def load_from_pytorch(self, pt_model: TorchGPT2, batch_size: int, history_length: int, action_length: int, flax_gpt2):
        # prepare dummy jax inputs and pass through the model
        dummy_history = jnp.zeros((batch_size, history_length, self.state_dim + self.action_dim))
        dummy_action = jnp.zeros((batch_size, action_length, self.action_dim))
        dummy_history_padding_mask = jnp.ones((batch_size, history_length))
        dummy_action_padding_mask = jnp.ones((batch_size, action_length))
        # params = self.init(key, dummy_history, dummy_action, dummy_history_padding_mask, dummy_action_padding_mask)

        # for name, module in pt_gpt2.named_modules():
        #     # for the linear layers, copy the weights and biases
        #     if isinstance(module, nn.Linear):
        #         kernel = module.weight.detach().cpu().numpy().T
        #         bias = module.bias.detach().cpu().numpy()
        #         params = params.set(f'{name}.kernel', kernel)
        #         params = params.set(f'{name}.bias', bias)
        #     # for gpt-2, use the huggingface conversion function

        params = modeling_flax_pytorch_utils.convert_pytorch_state_dict_to_flax(pt_model.state_dict(), flax_gpt2)
        params = {'params': params}

        out = self.apply(params, dummy_history, dummy_action, dummy_history_padding_mask, dummy_action_padding_mask, method=CustomFlaxGPT2Module.forward)
        return params
    
class FlaxGPT2(FlaxGPT2Model):
    module_class = CustomFlaxGPT2Module     
    
def main():
    # model = ParamTest()
    # rng = random.PRNGKey(0)
    # x = jnp.ones((1, 128))
    # params = model.init(rng, x)
    # y = model.apply(params, x)
    # print(y)

    config = GPT2Config(
        vocab_size=6 + 2,
        n_positions=550,
        n_embd=128,
        n_layer=3,
        n_head=4,
        n_inner=4 * 128,
        activation_function='gelu_new',
        resid_pdrop = 0.1,
        embd_pdrop = 0.1,
        attn_pdrop = 0.1,
        use_cache = False, # setting to true will be useful later for inference
    )
    model = FlaxGPT2(config=config, input_shape=(1, 550), state_dim=6, action_dim=2, output_dim=6, latent_dim=128, num_heads=4, num_layers=3)
    pt_model = TorchGPT2(state_dim=6, action_dim=2, output_dim=6, latent_dim=128, num_heads=4, num_layers=3)
    pt_model.load_state_dict(torch.load('/home/ubuntu/lecar-car/model_checkpoint.pth'))
    params = model.module.load_from_pytorch(pt_model, 256, 250, 50, model)
    
if __name__ == '__main__':
    main()
