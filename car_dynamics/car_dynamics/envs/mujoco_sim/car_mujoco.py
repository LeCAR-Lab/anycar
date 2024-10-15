from copy import deepcopy
from typing import Union
import gym
from gym import spaces
import numpy as np

from termcolor import colored
from car_dynamics.envs.mujoco_sim import World
from scipy.spatial.transform import Rotation as R

class MuJoCoCar(gym.Env):

    DEFAULT = {
        'max_step': 100,
        'dt': 0.02,
        'is_render': False,
        'delay': 0,
    }


    def __init__(self, config: dict):
        super(MuJoCoCar, self).__init__()
        
        self.config = deepcopy(self.DEFAULT)
        self.config.update(deepcopy(config))
        for key, value in self.config.items():
            assert key in self.DEFAULT, f'Bad key {key}'
            setattr(self, key, value)
    
        
        self.world = World({'is_render': self.is_render,})


        # Action space: target vel and steering
        self.action_space = spaces.Box(low=np.array([-1., -1.]), 
                                       high=np.array([1., 1.]), dtype=np.float32)
        
        self.observation_space = spaces.Box(
                ## x, y, psi, vx, vy, omega
                low=np.array([-np.inf] * 6), 
                high=np.array([np.inf] * 6), 
                dtype=np.float32,
        )

        self._step = None
        
        self.action_buffer = []
        
        self.target_velocity = 0

        self.wheelbase = 0.2965

        self.name = "mujoco"

        self.reset()


    def obs_state(self):
        """Return the 2D state of the car [x, y, psi, vx, vy, omega]

        Returns:
            np.array: [x, y, psi, vx, vy, omega]
        """
        
        quat = self.world.orientation
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        yaw = r.as_euler('zyx')[0]
        
        return np.array([
            self.world.pose[0],
            self.world.pose[1],
            yaw,
            self.world.lin_vel[0],
            self.world.lin_vel[1],
            self.world.ang_vel[2],
        ])
    

    def reset(self):
        self._step = 0.
        self.target_velocity = 0.
        self.world.reset()
        # for _ in range(20):
        #     self.step(np.array([-1, 0.]))
        self.action_buffer = []
        for _ in range(self.delay):
            self.action_buffer.append(np.array([0., 0.], dtype=np.float32))
            
        return self.obs_state()

    def reward(self,):
        return .0
    
    def step(self, action_):
        action_ = np.array(action_, dtype=np.float32)
        self.action_buffer.append(action_)
        self._step += 1
        action = self.action_buffer[0].copy()
        assert action.dtype == np.float32
        action = np.clip(action, self.action_space.low, self.action_space.high) #clip to -1, 1

        num_steps = int(self.dt / self.world.dt)

        # scale acceleration command 
        throttle = action[0] * self.world.max_throttle #scale it to real values
        steer = action[1] * self.world.max_steer  + self.world.steer_bias # scale it to real values
        for _ in range(num_steps):
            #calculate the target vel from throttle and previous velocity.
            self.target_velocity = throttle * self.world.dt + self.target_velocity
            action[0] = self.target_velocity
            action[1] = steer

            self.world.step(action)

        reward = self.reward()

        if self._step >= self.max_step:
            done = True
        else:
            done = False
            
        if self.is_render:
            self.render()

        self.action_buffer.pop(0)
        return self.obs_state(), reward, done, {}
    
    def render(self, mode='human'):
        if mode == 'human':
            self.world.render()

    def change_parameters(self, car_params: dict):
        self.world.change_parameters(car_params)

    # generate a new mass
    def generate_new_mass(self):   
        # print("[Warn] Car Mass Generation Not Defined for Simulator Type")
        default_mass = 3.794137 # base mujoco mass
        
        lower = default_mass * 0.7
        upper = default_mass * 1.3
        new_mass = np.random.uniform(lower, upper)

        return new_mass

    def generate_new_com(self):
        # print("[Warn] COM Generation Not Defined for Simulator Type")
        default_com = np.array([-0.02323112, -0.00007926,  0.09058852]) # base mujoco COM
        lower = default_com - 0.05 #5 cm range
        upper = default_com + 0.05

        new_com = np.random.uniform(lower, upper)

        return new_com
    
    def generate_new_friction(self):
        # print("[Warn] Friction Generation Not Defined for Simulator Type")

        default_friction = 1. #base mujoco friction
        lower = default_friction * 0.5
        upper = default_friction * 1.1
        friction = np.random.uniform(lower, upper)
        new_friction = np.array([friction, 0.005, 0.0001]) #static, torsional, rolling         

        return new_friction

    def generate_new_delay(self):
        lower = 0
        upper = 6
        new_delay = int(np.random.uniform(lower, upper))
        return new_delay
   
    def generate_new_max_throttle(self):
        # print("[Warn] Max Throttle Generation Not Defined for Simulator Type")
        lower = 2
        upper = 8
        max_thr = np.random.uniform(lower, upper)
        return max_thr
    
    def generate_new_max_steering(self):
        # print("[Warn] Max Steering Generation Not Defined for Simulator Type")
        lower = 0.15
        upper = 0.36
        max_steer = np.random.uniform(lower, upper)
        return max_steer
        # clip steering at random value
        # add bias to steering
        # 
        
    def generate_new_steering_bias(self):
        lower = 0.0
        upper = 0.01
        bias = np.random.uniform(lower, upper)
        return bias
        
    def generate_new_slope(self):
        # TODO not implemented right now
        print("[Warn] Slope Generation Not Defined for Simulator Type")
        pass
    
    
    