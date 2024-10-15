import jax
import time
import numpy as np
import jax.numpy as jnp
import optax
from jax import random
from rich.progress import track
from flax.training import train_state
from flax.core.frozen_dict import unfreeze
from termcolor import colored
from .networks import MLP
from .dbm import DynamicBicycleModel
from dataclasses import dataclass
from functools import partial

@dataclass
class AdaptDataset:
    state_list: jnp.ndarray
    action_list: jnp.ndarray
    
class ParamAdaptModel:
    """ This model will maintain a MLP and handle the online gradient descent process, 
            it implements physics informed neural network
    """

    def __init__(self,
                 dynamics: DynamicBicycleModel,
                 input_dims,
                 output_dims,
                 key1,
                 batch_size,
                 param_config,
                 learning_rate=.01, 
                 verbose = 0,
                 DEBUG = False,
                 model_dir = './',
                 ):
        ''' Maintains model save/load/train/evaluate/inference '''
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.key1 = key1
        self.verbose =verbose
        self.DEBUG = DEBUG
        self.dynamics = dynamics
        self._total_training_epoch = 0 
        self.param_config = param_config
        
        self.param_min = jnp.array([self.param_config[key][0] for key in self.param_config.keys()])
        self.param_max = jnp.array([self.param_config[key][1] for key in self.param_config.keys()])
    
    @property
    def train_input_dims(self):
        return self.input_dims

    @property
    def train_output_dims(self):
        return self.output_dims

    def get_params(self, params):
        return jax.nn.sigmoid(params)*(self.param_max - self.param_min) + self.param_min
    
    def mse_loss(self, params, state_action, next_state):
        # print(state_action.shape)
        params = self.get_params(params)
        # print(params)
        predict_next_state = self.dynamics.step_with_all_params(state_action, params)
        # print(predict_next_state - next_state)
        diff_state = predict_next_state - next_state
        diff_state = diff_state[:, [3, 4, 5]]
        diff_angle = (jnp.sin(predict_next_state[:, 2]) - jnp.sin(next_state[:, 2])) ** 2 + (jnp.cos(predict_next_state[:, 2]) - jnp.cos(next_state[:, 2])) ** 2
        return jnp.mean((diff_state) ** 2) * 1. + jnp.mean(diff_angle) * 1.

    @partial(jax.jit, static_argnums=(0,))     
    def update(self, params, x_a_batch, x_next_batch, opt_state):
        """ Update the model with a batch of data """
        loss, grads = jax.value_and_grad(self.mse_loss)(params, x_a_batch, x_next_batch)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss
    
    def adapt(self, 
              dataset: AdaptDataset, 
              init_params,
              epoches, 
              ):
        """ The dataset containts N paris of {state, }"""

        self.optimizer = optax.adam(self.learning_rate) 
        params = init_params.copy()
        opt_state = self.optimizer.init(params)
        # opt_state = self.optimizer.init(params)
        x_a_batch = jnp.concatenate([dataset.state_list[:-1], dataset.action_list[:-1]], axis=1)
        # print(dataset.action_list)
        x_next_batch = dataset.state_list[1:]
        loss_all = []
        for epoch in range(epoches):
            params, opt_state, loss = self.update(params, x_a_batch, x_next_batch, opt_state)
            loss_all.append(loss)
            
        # print(params)
        loss_mean = np.mean(loss_all)
        return self.get_params(params), {'loss': loss_mean, 'loss_all': loss_all}
    