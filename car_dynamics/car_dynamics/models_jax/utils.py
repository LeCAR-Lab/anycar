from typing import Any, Callable, Dict, Tuple
from termcolor import colored
import os
import jax.numpy as jnp

def create_missing_folder(directory, is_file_name=False):
    if is_file_name:
        paths = directory.split('/')
        # import pdb; pdb.set_trace()
        if directory[0] == '/':
            paths[0] = '/' + paths[0]
        if not os.path.exists(os.path.join(*paths[:-1])):
            os.makedirs(os.path.join(*paths[:-1]))
            print(colored(f"[INFO] Create path:{os.path.join(*paths[:-1])}", 'blue'))
    else:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(colored(f"[INFO] Create path:{directory}",'blue'))
            
def normalize_angle_tensor(x, idx):
    """ Normalize idx to np.cos(x[:,idx]) and np.sin(x[:,idx])
    """
    if len(x.shape) == 2:
        x_norm = jnp.zeros(x.shape[0], x.shape[1]+1)
        x_norm[:, :idx] = x[:, :idx]
        x_norm[:, idx] = jnp.cos(x[:, idx])
        x_norm[:, idx+1] = jnp.sin(x[:, idx])
        x_norm[:, idx+2:] = x[:, idx+1:]
        return x_norm
    elif len(x.shape) == 3:
        x_norm = jnp.zeros(x.shape[0], x.shape[1], x.shape[2]+1)
        x_norm[:, :, :idx] = x[:, :, :idx]
        x_norm[:, :, idx] = jnp.cos(x[:, :, idx])
        x_norm[:, :, idx+1] = jnp.sin(x[:, :, idx])
        x_norm[:, :, idx+2:] = x[:, :, idx+1:]
        return x_norm

def fold_angle_tensor(x, idx):
    if len(x.shape) == 2:
        x_fold = jnp.zeros(x.shape[0], x.shape[1]-1)
        x_fold[:, :idx] = x[:, :idx]
        x_fold[:, idx] = jnp.atan2(x[:, idx+1], x[:, idx])
        x_fold[:, idx+1:] = x[:, idx+2:]
        return x_fold
    elif len(x.shape) == 3:
        x_fold = jnp.zeros(x.shape[0], x.shape[1], x.shape[2]-1)
        x_fold[:, :, :idx] = x[:, :, :idx]
        x_fold[:, :, idx] = jnp.atan2(x[:, :, idx+1], x[:, :, idx])
        x_fold[:, :, idx+1:] = x[:, :, idx+2:]
        return x_fold