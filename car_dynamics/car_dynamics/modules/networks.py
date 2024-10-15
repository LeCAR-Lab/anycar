import jax.numpy as jnp
from flax import linen as nn


class AffineControlNN(nn.Module):
    input_dims: int 
    output_dims: int
    action_dims: int
    env_dims: int

    def setup(self):
        self.f = nn.Sequential([
            nn.Dense(128),
            nn.relu,
            nn.Dense(128),
            nn.relu,
            nn.Dense(self.output_dims)
        ])
        self.g = nn.Sequential([
            nn.Dense(128),
            nn.relu,
            nn.Dense(128),
            nn.relu,
            nn.Dense(self.output_dims * self.action_dims)
        ])

    def __call__(self, input):
        x, a = input[:, :-self.action_dims], input[:, -self.action_dims:]
        x, env_params = x[:, :-self.env_dims], x[:, -self.env_dims:]
        assert x.shape[1] == self.input_dims - self.env_dims - self.action_dims
        f_output = self.f(x)
        g_output = self.g(x)
        g_output_reshaped = g_output.reshape(-1, self.output_dims, self.action_dims)
        affine_term = jnp.matmul(g_output_reshaped, a[..., None]).squeeze(-1)
        return f_output + affine_term

    @property
    def dtype(self):
        return jnp.float32


class AffineInference(nn.Module):
    model: nn.Module


    @nn.compact
    def __call__(self, input):
        f = self.model.f
        g = self.model.g
        x, a = input[:, :-self.model.action_dims], input[:, -self.model.action_dims:]
        x, env_params = x[:, :-self.model.env_dims], x[:, -self.model.env_dims:]
        # __import__('pdb').set_trace()
        assert x.shape[1] == self.model.input_dims - self.model.env_dims - self.model.action_dims
        f_output = f(x)
        g_output = g(x)
        g_output = g_output.reshape(-1, self.model.output_dims, self.model.action_dims)
        return f_output, g_output

class AffineInference(nn.Module):
    model: nn.Module


    @nn.compact
    def __call__(self, input):
        f = self.model.f
        g = self.model.g
        x, a = input[:, :-self.model.action_dims], input[:, -self.model.action_dims:]
        x, env_params = x[:, :-self.model.env_dims], x[:, -self.model.env_dims:]
        # __import__('pdb').set_trace()
        assert x.shape[1] == self.model.input_dims - self.model.env_dims - self.model.action_dims
        f_output = f(x)
        g_output = g(x)
        g_output = g_output.reshape(-1, self.model.output_dims, self.model.action_dims)
        return f_output, g_output

class AffineControlEncNN(nn.Module):
    input_dims: int 
    output_dims: int
    action_dims: int
    env_dims: int
    repr_dims: int

    def setup(self):
        self.enc = nn.Sequential([
            nn.Dense(64),
            nn.relu,
            # nn.Dense(16),
            # nn.relu,
            nn.Dense(self.repr_dims),
        ])
        self.f = nn.Sequential([
            nn.Dense(256),
            nn.relu,
            nn.Dense(256),
            nn.relu,
            nn.Dense(self.output_dims)
        ])
        self.g = nn.Sequential([
            nn.Dense(256),
            nn.relu,
            nn.Dense(256),
            nn.relu,
            nn.Dense(self.output_dims * self.action_dims)
        ])

    def __call__(self, input):
        x, a = input[:, :-self.action_dims], input[:, -self.action_dims:]
        x, env_params = x[:, :-self.env_dims], x[:, -self.env_dims:]
        encoded_x = self.enc(env_params) # Encoding step
        assert encoded_x.shape[1] == self.repr_dims
        x = jnp.concatenate((x, encoded_x), axis=1)
        assert x.shape[1] == self.repr_dims + self.input_dims - self.env_dims -self.action_dims
        
        f_output = self.f(x) # Feed encoded_x to f
        g_output = self.g(x) # Feed encoded_x to g
        g_output_reshaped = g_output.reshape(-1, self.output_dims, self.action_dims)
        affine_term = jnp.matmul(g_output_reshaped, a[..., None]).squeeze(-1)
        return f_output + affine_term

class AffineEncInference(nn.Module):
    model: nn.Module

    @nn.compact
    def __call__(self, input):
        f = self.model.f
        g = self.model.g
        enc = self.model.enc
        x, a = input[:, :-self.model.action_dims], input[:, -self.model.action_dims:]
        x, env_params = x[:, :-self.model.env_dims], x[:, -self.model.env_dims:]
        assert x.shape[1] == self.model.input_dims - self.model.env_dims - self.model.action_dims
        encoded_x = enc(env_params) # Encoding step
        assert encoded_x.shape[1] == self.model.repr_dims
        x = jnp.concatenate((x, encoded_x), axis=1)
        assert x.shape[1] == self.model.repr_dims + self.model.input_dims - self.model.env_dims - self.model.action_dims

        f_output = f(x)
        g_output = g(x)
        g_output = g_output.reshape(-1, self.model.output_dims, self.model.action_dims)
        return f_output, g_output


class AdaptationNN(nn.Module):
    input_dims: int
    output_dims: int

    def setup(self):
        self.enc = nn.Sequential([
            nn.Dense(128),
            nn.relu,
            nn.Dense(128),
            nn.relu,
            nn.Dense(self.output_dims),
        ])

    def __call__(self, env_params):
        assert env_params.shape[1] == self.input_dims
        encoded_x = self.enc(env_params) # Encoding step
        return encoded_x

class Flatten(nn.Module):
    def __call__(self, x):
        return x.reshape((x.shape[0], -1))

class AddChannelDim(nn.Module):
    def __call__(self, x):
        return jnp.expand_dims(x, axis=-1)

class AdaptationCNN(nn.Module):
    input_dims: int
    output_dims: int

    def setup(self):
        self.enc = nn.Sequential([
            nn.Dense(32),
            nn.relu,
            nn.Dense(32),
            AddChannelDim(),
            nn.Conv(features=32, kernel_size=(8,), strides=(4,)),
            nn.relu,
            nn.Conv(features=32, kernel_size=(8,), strides=(4,)),
            nn.relu,
            Flatten(),
            nn.Dense(self.output_dims),
        ])

    def __call__(self, env_params):
        assert env_params.shape[1] == self.input_dims
        # x = jnp.expand_dims(x, axis=-1)
        encoded_x = self.enc(env_params) # Encoding step
        # __import__('pdb').set_trace()
        return encoded_x

class AdaptationDynamicsNN(nn.Module):
    input_dims: int 
    output_dims: int
    action_dims: int
    env_dims: int
    repr_dims: int
    enc: nn.Module
    model: nn.Module
    repr_min: jnp.ndarray
    repr_max: jnp.ndarray

    @nn.compact
    def __call__(self, input):
        # __import__('pdb').set_trace()
        enc = self.enc.enc
        f = self.model.f
        g = self.model.g

        x, a = input[:, :-self.action_dims], input[:, -self.action_dims:]
        x, env_params = x[:, :-self.env_dims], x[:, -self.env_dims:]
        assert x.shape[1] == self.input_dims - self.action_dims - self.env_dims
        encoded_x = enc(env_params) # Encoding step
        encoded_x = encoded_x * (self.repr_max - self.repr_min + 1e-8) + self.repr_min
        assert encoded_x.shape[1] == self.repr_dims
        # __import__('pdb').set_trace()
        x = jnp.concatenate((x, encoded_x), axis=1)
        assert x.shape[1] == self.repr_dims + self.input_dims - self.env_dims -self.action_dims
        
        f_output = f(x) # Feed encoded_x to f
        g_output = g(x) # Feed encoded_x to g
        g_output_reshaped = g_output.reshape(-1, self.output_dims, self.action_dims)
        affine_term = jnp.matmul(g_output_reshaped, a[..., None]).squeeze(-1)
        return f_output + affine_term

class AdaptationDynamicsNNInference(nn.Module):
    enc: nn.Module
    model: nn.Module
    repr_min: jnp.ndarray
    repr_max: jnp.ndarray
    input_dims: int 
    output_dims: int
    action_dims: int
    env_dims: int
    repr_dims: int

    @nn.compact
    def __call__(self, input):
        # __import__('pdb').set_trace()
        enc = self.enc.enc
        f = self.model.f
        g = self.model.g

        x, a = input[:, :-self.action_dims], input[:, -self.action_dims:]
        x, env_params = x[:, :-self.env_dims], x[:, -self.env_dims:]
        assert x.shape[1] == self.input_dims - self.action_dims - self.env_dims
        encoded_x = enc(env_params) # Encoding step
        # print(f"ENCODED X: {encoded_x}")
        encoded_x = encoded_x * (self.repr_max - self.repr_min + 1e-8) + self.repr_min
        assert encoded_x.shape[1] == self.repr_dims
        x = jnp.concatenate((x, encoded_x), axis=1)
        assert x.shape[1] == self.repr_dims + self.input_dims - self.env_dims -self.action_dims

        f_output = f(x)
        g_output = g(x)
        g_output = g_output.reshape(-1, self.model.output_dims, self.model.action_dims)
        return f_output, g_output


class PretrainedEnc(nn.Module):
    enc: nn.Module

    @nn.compact
    def __call__(self, input):
        enc = self.enc.enc
        encoded = enc(input)
        return encoded 
