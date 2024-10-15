from copy import deepcopy
import os
import jax
import jax.numpy as jnp
import time
import orbax
import numpy as np
from car_foundation import CAR_FOUNDATION_MODEL_DIR
from car_foundation.jax_models import JaxTransformerDecoder
from termcolor import colored
from flax.training import orbax_utils
from functools import partial

def align_yaw(yaw_1, yaw_2):
    d_yaw = yaw_1 - yaw_2
    d_yaw_aligned = jnp.arctan2(jnp.sin(d_yaw), jnp.cos(d_yaw))
    return d_yaw_aligned + yaw_2

class DynamicsJax:
    
    def __init__(self, params: dict):
        self.params = deepcopy(params)
        
        model_path = self.params.get("model_path", "")
        
        self.key = jax.random.PRNGKey(123)
        state_dim = 6
        action_dim = 2
        latent_dim = 64
        num_heads = 4
        num_layers = 2
        dropout = 0.1
        history_length = 250
        prediction_length = 50
        batch_size = 512
        
        self.model = JaxTransformerDecoder(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, dropout, history_length, prediction_length, jnp.bfloat16)
        
        jax_history_input = jnp.ones((batch_size, history_length, state_dim + action_dim), dtype=jnp.float32)
        jax_history_mask = jnp.ones((batch_size, history_length * 2 - 1), dtype=jnp.float32)
        jax_prediction_input = jnp.ones((batch_size, prediction_length, action_dim), dtype=jnp.float32)
        jax_prediction_mask = jnp.ones((batch_size, prediction_length), dtype=jnp.float32)
        
        self.key, key2 = jax.random.split(self.key, 2)
        self.var = self.model.init(key2, jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask)
        
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        print(colored(f"Loading model from: {model_path}", "green"))
        raw_restored = orbax_checkpointer.restore(model_path)
        
        # Print the model structure, and count the number of parameters
        # orbax_utils.print_model_structure(raw_restored['model'])
        # orbax_utils.count_parameters(raw_restored['model'])
        param_count = sum(x.size for x in jax.tree_leaves(raw_restored['model']['params']))
        # print(f"model structu: {raw_restored['model']}")
        print(f"Number of parameters: {param_count}")
        # import pdb; pdb.set_trace()
        
        # self.var['params'] = raw_restored['params']
        # self.input_mean = jnp.array([3.3083595e-02, 1.4826456e-04, 1.8982769e-03, 1.6544139e+00, 5.5305376e-03, 9.5738873e-02])
        # self.input_std = jnp.array([0.01598073, 0.00196785, 0.01215522, 0.7989133 , 0.09668902 ,0.608985  ])
        
        self.var['params'] = raw_restored['model']['params']
        self.input_mean = jnp.array(raw_restored['input_mean'])
        self.input_std = jnp.array(raw_restored['input_std'])
        
        print("input_mean", self.input_mean)
        print("input_std", self.input_std)
        
        
    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, history: jax.Array, state: jax.Array, action: jax.Array):
        st_nn_dyn = time.time()

        # convert pose state to delta
        original_yaw = history[:, :-1, 2]
        batch_tf = history
        batch_tf = batch_tf.at[:, 1:, :3].set(batch_tf[:, 1:, :3] - batch_tf[:, :-1, :3])
        batch_tf = batch_tf.at[:, 1:, 2].set(align_yaw(batch_tf[:, 1:, 2], 0.0))
        # rotate dx, dy into body frame
        batch_tf_x = batch_tf[:, 1:, 0] * jnp.cos(original_yaw) + batch_tf[:, 1:, 1] * jnp.sin(original_yaw)
        batch_tf_y = -batch_tf[:, 1:, 0] * jnp.sin(original_yaw) + batch_tf[:, 1:, 1] * jnp.cos(original_yaw)
        batch_tf = batch_tf.at[:, 1:, 0].set(batch_tf_x)
        batch_tf = batch_tf.at[:, 1:, 1].set(batch_tf_y)
        batch_tf = batch_tf.at[:, :, :6].set((batch_tf[:, :, :6] - self.input_mean) / self.input_std)

        x = batch_tf[:, 1:, :]
        key, key2 = jax.random.split(key, 2)
        # print("Shape inside", action.shape)
        y_pred = ( self.model.apply(self.var, x, action, action_padding_mask=None, rngs=key2, deterministic=True) * self.input_std + self.input_mean ) #* self.input_std + self.input_mean

        last_pose = history[:, -1, :3]
        for i in range(y_pred.shape[1]):
            # rotate dx, dy back to world frame
            y_pred_x = y_pred[:, i, 0] * jnp.cos(last_pose[:, 2]) - y_pred[:, i, 1] * jnp.sin(last_pose[:, 2])
            y_pred_y = y_pred[:, i, 0] * jnp.sin(last_pose[:, 2]) + y_pred[:, i, 1] * jnp.cos(last_pose[:, 2])
            y_pred = y_pred.at[:, i, 0].set(y_pred_x)
            y_pred = y_pred.at[:, i, 1].set(y_pred_y)
            # accumulate the poses
            y_pred = y_pred.at[:, i, :3].set(y_pred[:, i, :3] + last_pose)
            y_pred = y_pred.at[:, i, 2].set( align_yaw(y_pred[:, i, 2], 0.0) )
            last_pose = y_pred[:, i, :3]
            
        print("NN Inference Time", time.time() - st_nn_dyn)
        return y_pred
        