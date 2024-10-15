from copy import deepcopy
import gym
from gym import spaces
import numpy as np
from ..kinematic_bicycle_model import BikePID

class Navigation2(gym.Env):
    '''We assume the only effect of box/chair on the car is changing the Kp,Kd,max_vel,max_steer,
        This env is used for dynamics learning, no hazard'''

    #### REAL SCALE FloorB NSH @CMU #
    #   |<----- 1m ---->|           #
    #   |        ^      |           #
    #   |        |      |           #
    #   |        |      |           #
    #   |        |      |           #
    #   |        | 2m   |           #
    #   |        |      |           #
    #   |        |      |           #
    #   |        |      |           #
    #   |        v      |           #
    #   |---------------|           #
    # ###############################

    _max_x = 1.0 
    _max_y = 2.0


    
    DEFAULT = {
        'field_max_x': 1.0,
        'field_max_y': 2.0,

        ### RL Penalty
        'reward_steer': 0.01,
        'reward_speed': 0.01,
        'reward_distance': 1.,
        'reward_goal': 10.,
        'reward_out_of_boundary': 5.,
        'reward_orientation': 0.1,

        ### Env Configurations
        'env_Kp': [1., 8.],
        'env_Kd': [.01, .02],
        'env_max_vel': [4., 8.],
        'env_max_steer': [.3, .4],

        ### Task
        'init_pos': [0., 0.],
        'init_yaw': None,
        'init_vel': None,
        'goal_pos': [0., 4.],
        'max_step': 100,
        'dt': 0.05,
        'proj_steer': .34,
        'shift_steer': 0.,
    }


    def __init__(self, config):
        super(Navigation2, self).__init__()
        
        self.sim = BikePID()

        self.config = deepcopy(self.DEFAULT)
        self.config.update(deepcopy(config))
        for key, value in self.config.items():
            assert key in self.DEFAULT, f'Bad key {key}'
            setattr(self, key, value)

        # Action space: move in x and y direction [-1, 1]
        self.action_space = spaces.Box(low=np.array([-1., -1.]), 
                                       high=np.array([1., 1.]), dtype=np.float32)
        
        # State space: position in x and y direction
        # self.observation_space = spaces.Box(
        #         low=np.array([-np.inf, -np.inf, -np.inf, -np.inf,-np.inf,-np.inf]), 
        #         high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]), 
        #         dtype=np.float32,
        # )
        self.observation_space = spaces.Box(
                ## x, y, yaw_x, yaw_y, vel, Fx, Fy
                low=np.array([-np.inf] * (5+4)), 
                high=np.array([np.inf] * (5+4)), 
                dtype=np.float32,
        )

        self.pos = None
        self.yaw = None
        self.vel = None
        self._step = None

        self.reset()

    def _dist_goal(self):
        assert self.pos is not None
        assert self.goal is not None
        return np.linalg.norm(self.pos - self.goal) 

    def obs_state(self):
        return np.array([
            self.pos[0],
            self.pos[1],
            self.yaw,
            self.vel,
        ]).copy()
        
    def obs(self):
        return np.array([
            self.pos[0],
            self.pos[1],
            np.cos(self.yaw),
            np.sin(self.yaw),
            self.vel,
            self.Kp,
            self.Kd,
            self.max_steer,
            self.max_vel,
        ]).copy()

    def reset(self):
        # task        self.pos = deepcopy(self.init_pos)
        self.pos = np.array(self.init_pos)
        self.goal = np.array(self.goal_pos)
        if self.init_yaw is None:
            self.yaw = np.random.uniform(-np.pi, np.pi)
        else:
            self.yaw = self.init_yaw

        if self.init_vel is None:
            self.vel = 0
        else:
            self.vel = self.init_vel

        self._step = 0
        self.last_dist_goal = self._dist_goal()

        # env params
        self.Kp = np.random.uniform(self.env_Kp[0], self.env_Kp[1])
        self.Kd = np.random.uniform(self.env_Kd[0], self.env_Kd[1])
        self.max_steer = np.random.uniform(self.env_max_steer[0], self.env_max_steer[1])
        self.max_vel = np.random.uniform(self.env_max_vel[0], self.env_max_vel[1])

        self.sim.reset()

        return self.obs()

    def reward(self,):
        return .0
    
    def step(self, action_):

        # normalize action to [-1, 1.]
        self._step += 1
        action = action_ + .0
        action[0] = max(min(action[0], 1.), -1.)
        action[1] = max(min(action[1], 1.), -1.)

        pos_x, pos_y, self.yaw, self.vel = self.sim.step(
            pos_x = self.pos[0],
            pos_y = self.pos[1],
            psi = self.yaw,
            vel = self.vel,
            target_vel = action[0],
            steer = action[1],
            dt = self.dt,
            # max_steer = self.max_steer + np.random.uniform(-.2, .2),
            max_steer = self.max_steer,
            MaxVel= self.max_vel,
            Kp = self.Kp,
            Kd = self.Kd,
            proj_steer = self.proj_steer,
            shift_steer =self.shift_steer,
        )

        self.pos = np.array([pos_x, pos_y]) 


        reward = self.reward()

        if self._step >= self.max_step:
            done = True
        else:
            done = False

        return self.obs(), reward, done, {}
    
    def render(self, mode='human'):
        ...


