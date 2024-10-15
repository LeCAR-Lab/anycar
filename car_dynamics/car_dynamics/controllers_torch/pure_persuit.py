import numpy as np
from scipy.spatial.transform import Rotation as R

class PurePersuitParams:
    wheelbase = 0.31
    max_speed = 8.
    max_steering_angle = 0.34
    target_vel = 2.
    mode = 'vel'
    kp = 1
    kd = 0.1
    
    
class PurePersuitController:
    
    def __init__(self, params: PurePersuitParams):
        self.params = params
        self.last_vel_error = 0.
    
    def step(self, pose, lin_vel, quat, target_pos_list: list):
        """_summary_

        Args:
            target_pos_list (list): list of target positions to follow
        Returns:
            np.array: [vel, steering_angle] normalized to [-1, 1] 
        """
        lookahead_x, lookahead_y, _, _ = target_pos_list[-1]
        # transform the lookahead point to the car's frame
        x, y, _ = pose
        x_lookahead = lookahead_x - x
        y_lookahead = lookahead_y - y
        print("Distance", np.linalg.norm([x_lookahead, y_lookahead]))
        # rotate the lookahead point to the car's frame
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        yaw = r.as_euler('zyx')[0]
        x_lookahead_rot = np.cos(yaw) * x_lookahead + np.sin(yaw) * y_lookahead
        y_lookahead_rot = -np.sin(yaw) * x_lookahead + np.cos(yaw) * y_lookahead
        lookahead_pt = np.array([x_lookahead_rot, y_lookahead_rot])
        
        lookahead_angle = np.arctan2(lookahead_pt[1], lookahead_pt[0])
        lookahead_distance = np.sqrt(pow(lookahead_pt[1], 2) + pow(lookahead_pt[0], 2))
        steer_rad = np.arctan((2 * self.params.wheelbase * np.sin(lookahead_angle)) / lookahead_distance)
        steer = np.clip(steer_rad/self.params.max_steering_angle, -1., 1.)
        # target_vel = self.params.target_vel
        target_vel = target_pos_list[0][3]
        
        if self.params.mode == 'vel':
            half_speed = self.params.max_speed/2
            cmd_speed = (target_vel - half_speed) / half_speed
            cmd_speed = np.clip(cmd_speed, -1., 1.)
            return np.array([cmd_speed, steer])
        elif self.params.mode == 'throttle':
            cmd_throttle = self.params.kp * (target_vel - lin_vel[0]) + self.params.kd * ((target_vel - lin_vel[0]) - self.last_vel_error)
            self.last_vel_error = target_vel - lin_vel[0]
            cmd_throttle = np.clip(cmd_throttle, -1., 1.)
            return np.array([cmd_throttle, steer])
        else:
            raise ValueError(f"Invalid mode: {self.params.mode}")
            
