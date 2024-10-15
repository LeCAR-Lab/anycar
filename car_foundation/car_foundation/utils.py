from collections import OrderedDict
import numpy as np
import torch
import jax.numpy as jnp

class HyperparameterManager:
    def __init__(self):
        self.hyperparameters = OrderedDict()

    def get_hyperparameter(self, key):
        return self.hyperparameters[key]

    def get_hyperparameters(self):
        return self.hyperparameters

    def set_hyperparameter(self, key, value):
        self.hyperparameters[key] = value

    def set_hyperparameters(self, hyperparameters):
        self.hyperparameters = hyperparameters

    def __str__(self):
        return str(self.hyperparameters)

    def __repr__(self):
        return str(self.hyperparameters)
    
    def __getitem__(self, key):
        return self.hyperparameters[key]
    
    def __setitem__(self, key, value):
        self.hyperparameters[key] = value
    
    def __iter__(self):
        return iter(self.hyperparameters)
    
    def __len__(self):
        return len(self.hyperparameters)
    
    def __contains__(self, key):
        return key in self.hyperparameters
    
    def __delitem__(self, key):
        del self.hyperparameters[key]
        

def quaternion_to_euler(q):
    # Normalize quaternion
    norm = np.linalg.norm(q, axis=1)[:, np.newaxis]
    q = q / norm
    
    # Extract the values from Q
    q_w, q_x, q_y, q_z = q[:,0], q[:,1], q[:,2], q[:,3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (q_w * q_x + q_y * q_z)
    cosr_cosp = 1 - 2 * (q_x**2 + q_y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (q_w * q_y - q_z * q_x)
    pitch = np.where(np.abs(sinp) >= 1,
                    np.sign(sinp) * np.pi / 2,  # use 90 degrees if out of range
                    np.arcsin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (q_w * q_z + q_x * q_y)
    cosy_cosp = 1 - 2 * (q_y**2 + q_z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def generate_subsequences(input_tensor):
    """
    Generates subsequences from the input tensor with increasing length,
    pads them to full length, and generates a padding mask.

    Args:
    input_tensor (torch.Tensor): Tensor of shape (N, S, E) where
        N is the batch size,
        S is the sequence length,
        E is the vector dimension.

    Returns:
    tuple: A tuple containing:
        - output_tensors (torch.Tensor): Tensor of shape (N * S, S, E)
        where each sub-tensor includes subsequences padded to full length.
        - mask (torch.Tensor): Float32 mask of shape (N * S, S) indicating
        paddings (-inf for padding, 0 for data).
    """
    N, S, E = input_tensor.shape
    # Initialize a tensor to hold the padded subsequences
    output_tensors = torch.zeros(N, S, S, E, device=input_tensor.device, dtype=input_tensor.dtype)
    mask = torch.fill_(torch.zeros(N, S, S, dtype=torch.float32, device=input_tensor.device), float('-inf'))

    # Loop over each possible subsequence length
    for i in range(S):
        output_tensors[:, i, :i+1, :] = input_tensor[:, :i+1, :]
        mask[:, i, :i+1] = 0.0

    return output_tensors.view(N * S, S, E), mask.view(N * S, S)

def generate_subsequences_hf(input_tensor):
    """
    Generates subsequences from the input tensor with increasing length,
    pads them to full length, and generates a padding mask.
    HuggingFace convention: mask is 1 for data and 0 for padding.

    Args:
    input_tensor (torch.Tensor): Tensor of shape (N, S, E) where
        N is the batch size,
        S is the sequence length,
        E is the vector dimension.

    Returns:
    tuple: A tuple containing:
        - output_tensors (torch.Tensor): Tensor of shape (N * S, S, E)
        where each sub-tensor includes subsequences padded to full length.
        - mask (torch.Tensor): Float32 mask of shape (N * S, S) indicating
        paddings (0 for padding, 1 for data).
    """
    N, S, E = input_tensor.shape
    # Initialize a tensor to hold the padded subsequences
    output_tensors = torch.zeros(N, S, S, E, device=input_tensor.device, dtype=input_tensor.dtype)
    mask = torch.zeros(N, S, S, dtype=torch.float32, device=input_tensor.device)

    # Loop over each possible subsequence length
    for i in range(S):
        output_tensors[:, i, :i+1, :] = input_tensor[:, :i+1, :]
        mask[:, i, :i+1] = 1.0

    return output_tensors.view(N * S, S, E), mask.view(N * S, S)

def align_yaw(yaw_1, yaw_2):
    d_yaw = yaw_1 - yaw_2
    d_yaw_aligned = torch.atan2(torch.sin(d_yaw), torch.cos(d_yaw))
    return d_yaw_aligned + yaw_2

def align_yaw_jax(yaw_1, yaw_2):
    d_yaw = yaw_1 - yaw_2
    d_yaw_aligned = jnp.atan2(jnp.sin(d_yaw), jnp.cos(d_yaw))
    return d_yaw_aligned + yaw_2