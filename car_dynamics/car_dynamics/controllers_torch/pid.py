
import numpy as np
from car_dynamics.controllers_torch import BaseController


class PIDController( BaseController ):
    def __init__(self,
                 kp,
                 kd,
                 ki,
        ):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.integral = 0.
        self.prev_error = 0.
        
    def __call__(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        steering = self.kp * error + self.kd * derivative + self.ki * self.integral
        # steering = self.kp * error
        
        steering = np.clip(steering, -1., 1.)
        return steering
    