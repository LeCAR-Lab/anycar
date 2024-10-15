from copy import deepcopy
import gym
from gym import spaces
import numpy as np
from car_dynamics.models_jax import DynamicBicycleModel, CarState, CarAction
from termcolor import colored
from scipy.spatial.transform import Rotation as R


class Car2D(gym.Env):
    '''WARNING: There is no lin_acc information in this sim
    '''

    DEFAULT = {
        'max_step': 100,
        'delay': 0,
    }


    def __init__(self, config: dict, dynamics: DynamicBicycleModel):
        super(Car2D, self).__init__()
        
        self.config = deepcopy(self.DEFAULT)
        self.config.update(deepcopy(config))
        for key, value in self.config.items():
            assert key in self.DEFAULT, f'Bad key {key}'
            setattr(self, key, value)
            
        self.sim = dynamics


        # Action space: move in x and y direction [-1, 1]
        self.action_space = spaces.Box(low=np.array([-1., -1.]), 
                                       high=np.array([1., 1.]), dtype=np.float32)
        
        self.observation_space = spaces.Box(
                ## x, y, yaw_x, yaw_y, vel, Fx, Fy
                low=np.array([-np.inf] * 6), 
                high=np.array([np.inf] * 6), 
                dtype=np.float32,
        )

        self._step = None
        
        self.action_buffer = list()
        self.reset()


    def obs_state(self):
        return np.array([self.state.x, self.state.y, self.state.psi, self.state.vx, self.state.vy, self.state.omega])
    
    @property
    def car_pos(self):
        return np.array([
            self.state.x, 
            self.state.y, 
            0.
        ])
        
    @property
    def car_lin_vel(self):
        return np.array([
            self.state.vx, 
            self.state.vy, 
            0.
        ])
    
    @property
    def car_ang_vel(self):
        return np.array([
            0., 
            0., 
            self.state.omega
        ])
    
    @property
    def car_orientation(self):
        ori1 = np.array([
            np.cos(self.state.psi/2),
            0., 
            0., 
            np.sin(self.state.psi/2), 
        ])
        r = R.from_euler('xyz', [0., 0., self.state.psi])
        ori2 = r.as_quat()
        ori2 = np.array([ori2[3], ori2[0], ori2[1], ori2[2]])
        assert np.allclose(ori1, ori2)
        return ori1
        
    @property
    def car_rpy(self):
        return np.array([0., 0., self.state.psi])
    
    @property
    def car_lin_acc(self):
        return np.array([0., 0., 0.])
    
    @property
    def wheelbase(self):
        return (self.sim.params.LF + self.sim.params.LR)


    def reset(self):
        self.state = CarState(
            x=0,
            y=0,
            psi=0.,
            vx=0.,
            vy=0,
            omega=0.0,
        )

        self._step = 0
        self.sim.reset()
        
        self.action_buffer = []
        for _ in range(self.delay):
            self.action_buffer.append(np.array([0., 0.], dtype=np.float32))

        return self.obs_state()

    def reward(self,):
        return .0
    
    def step(self, action_):
        action_ = np.array(action_, dtype=np.float32)
        self.action_buffer.append(action_)
        # normalize action to [-1, 1.]
        self._step += 1
        action = self.action_buffer[0].copy()
        assert action.dtype == np.float32
        action = np.clip(action, self.action_space.low, self.action_space.high)

        action = CarAction(
            target_vel = action[0],
            target_steer = action[1],
        )
        
        self.state = self.sim.step_gym(self.state, action)
        reward = self.reward()

        if self._step >= self.max_step:
            done = True
        else:
            done = False
            
        self.action_buffer.pop(0)

        return self.obs_state(), reward, done, {}
    
    def render(self, mode='human'):
        ...


    @property
    def pose(self):
        return self.car_pos
    
    @property
    def rpy(self):
        return self.car_rpy