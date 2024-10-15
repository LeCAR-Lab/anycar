import numpy as np
from dataclasses import dataclass
from typing import Union, List

@dataclass
class CarEnvParams:
    name: str
    mass: float
    friction: Union[float, List[float]]
    render: bool
    delay: int
    max_throttle: float
    max_steer: float
    steer_bias: float
    wheelbase: float
    com: Union[float, List[float]]
    

def make_env(params: CarEnvParams):
    
    if params.name == 'car-numeric-2d':
        from car_dynamics.envs.numeric_sim import Car2D
        from car_dynamics.models_jax import DynamicBicycleModel, DynamicParams
        
        dynamic_param = DynamicParams(
            num_envs = 1,
            MASS = params.mass,
            mu = params.friction,
            Ta = params.max_throttle,
            Sa = params.max_steer,
            Sb = params.steer_bias,
            LF = params.wheelbase * params.com,
            LR = params.wheelbase * (1 - params.com),
        )
        dynamics = DynamicBicycleModel(dynamic_param)
        dynamics.reset()
        env = Car2D(
            {
                'delay': params.delay,
            }, 
            dynamics
        )
        return env
    else:
        raise ValueError(f'Unknown env {params.name}')