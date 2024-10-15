import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import splprep, splev
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt 
import math

class RandWalkController:
    def __init__(self, totaltime):
        self.totaltime = totaltime

        self.lowervel = -1
        self.uppervel = 1
        self.lowersteer = -1
        self.uppersteer = 1

        self.generate_target_velocities()
        self.generate_target_steering()

        self.target_pos = [0, 0]
        self.name = "random_walk"
        
    def generate_target_velocities(self):
        samples = self.totaltime // 100
        mean = 0
        std_dev = 0.5
        sampled_vels = np.random.normal(mean, std_dev, samples)
        # print(min(sampled_vels), max(sampled_vels ))
        sampled_vels = np.clip(sampled_vels, self.lowervel, self.uppervel)
        x = np.linspace(0, self.totaltime, samples)
        spline = make_interp_spline(x, sampled_vels)
        x_total = np.arange(0, self.totaltime)
        clipped_spline = np.clip(spline(x_total), self.lowervel, self.uppervel)
        self.target_velocities = clipped_spline
        # plt.plot(self.target_velocities)
        # plt.show()

    def generate_target_steering(self):
        samples = self.totaltime // 100
        sampled_steers = np.random.uniform(self.lowersteer, self.uppersteer, samples)
        x = np.linspace(0, self.totaltime, samples)
        spline = make_interp_spline(x, sampled_steers)
        x_total = np.arange(0, self.totaltime)
        clipped_spline = np.clip(spline(x_total), self.lowersteer, self.uppersteer)
        self.target_steer = clipped_spline

    def get_control(self, t, world, path):
        throttle = self.target_velocities[t]
        steer = self.target_steer[t]
        return [throttle, steer]
    
    def reset_controller(self):
        self.generate_target_velocities()
        self.generate_target_steering()