import jax.numpy as jnp
from flax import linen as nn

class MLP(nn.Module):
    input_dims: int 
    output_dims: int

    def setup(self):
        self.f = nn.Sequential([
            nn.Dense(128),
            nn.relu,
            nn.Dense(128),
            nn.relu,
            nn.Dense(self.output_dims)
        ])

    def __call__(self, input):
        return self.f(input)
