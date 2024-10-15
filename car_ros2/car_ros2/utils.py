# import rclpy
from datetime import datetime, timedelta
from car_dynamics.models_jax import DynamicParams
from car_dynamics.controllers_jax import MPPIParams
from car_dynamics.envs import CarEnvParams
from typing import Union, List

def load_env_params_numeric() -> CarEnvParams:
    return CarEnvParams(
        mass = 4.,
        friction=.8,
        name = 'car-numeric-2d',
        render = False,
        delay=0,
        max_throttle=8.,
        max_steer=0.36,
        steer_bias=0.025,
        wheelbase=0.21,
        com=0.48,
        
    )
    
def load_env_params_mujoco() -> CarEnvParams:
    return CarEnvParams(
        mass = 3.794,
        friction=.8,
        name = 'car-mujoco',
        # render = False,
        render = True,
        delay=0,
        max_throttle=8.,
        max_steer=0.36,
        steer_bias=0.0,
        wheelbase=0.,
        com=0.,
    )
    
def load_env_params_isaacsim() -> CarEnvParams:
    return CarEnvParams(
        mass = 3.794,
        friction=.8,
        name= 'car-isaacsim',
        # render = False,
        render = True,
        delay=0,
        max_throttle=16.,
        max_steer=0.36,
        steer_bias=0.0,
        wheelbase=0.,
        com=0.,
    )
    
def load_env_params_unity() -> CarEnvParams:
    return CarEnvParams(
        mass = 3.794,
        friction=.8,
        # name = 'car-numeric-2d',
        # name = 'car-mujoco',
        name= 'car-unity-beta',
        # name= 'car-isaacsim',
        render = False,
        # render = True,
        delay=0,
        max_throttle=16.,
        max_steer=0.36,
        steer_bias=0.0,
        wheelbase=0.,
        com=0.,
    )

def load_dynamic_params() -> DynamicParams:
    num_envs = 200
    LF = 0.12
    LR = 0.08
    DT = 0.02
    Sa = 0.36
    Sb = 0.0
    Ta = 4.0
    Tb = 0.0
    mu = 0.8
    MASS = 1.8
    return DynamicParams(
        num_envs=num_envs, LF=LF, LR=LR, DT=DT, Sa=Sa, Sb=Sb, Ta=Ta, Tb=Tb, mu=mu, MASS=MASS
    )
    

def load_mppi_params() -> MPPIParams:
    
    return MPPIParams(
        spline_order=2,
        sigma=0.05,
        gamma_sigma=0.0,
        gamma_mean=1.0,
        discount=1.0,
        sample_sigma=1.0,
        lam=0.1,
        n_rollouts=600,        
        a_min=[-1, -1.], # first dim steer, 2nd throttle
        a_max=[1., 1.],
        a_mag=[1., 1.],
        a_shift=[0., 0.],
        delay=0,
        len_history=251,
        debug=False,
        fix_history=False,
        num_obs=6,
        num_actions=2,
        num_intermediate=7,
        h_knot=8,
        smooth_alpha=1.0,
        dynamics="transformer-jax",
        dual=True, 
        # dynamics="dbm",
        # dual=False, 
    )
    

def rospy_time_to_datetime(ros_time):
    # if ros_time is iterable, return a list of datetime objects
    if hasattr(ros_time, '__iter__'):
        return [rospy_time_to_datetime(t) for t in ros_time]
    else:
        secs = ros_time.sec
        nsecs = ros_time.nanosec
        dt = datetime.fromtimestamp(secs + nsecs * 1e-9)
        return f"{dt}"

def rospy_time_datatime_float(ros_time) -> Union[float, List[float]]:
    # if ros_time is iterable, return a list of datetime.timestamp() floats
    if hasattr(ros_time, '__iter__'):
        return [rospy_time_datatime_float(t) for t in ros_time]
    else:
        secs = ros_time.sec
        nsecs = ros_time.nanosec
        dt = datetime.fromtimestamp(secs + nsecs * 1e-9)
        start_time = datetime(1970, 1, 1)
        delta_time = (dt - start_time).total_seconds()
        return delta_time
