import numpy as np

class CarDataset:
    def __init__(self) -> None:
        self.reset_logs()  
        self.car_params = {"wheelbase": None, 
                           "mass": None, 
                           "com": None, 
                           "friction": None,  
                           "delay": 0, 
                           'max_throttle': None, 
                           'max_steer': None,
                           'steer_bias': None,
                           "sim": None,}

    
    def reset_logs(self):
        self.data_logs = {   "steer": [],
                        "throttle": [],
                        "xpos_x": [],  # X component of position
                        "xpos_y": [],  # Y component of position
                        "xpos_z": [],  # Z component of position
                        "xori_w": [],  # W compoenent of orientation (Quaternion)
                        "xori_x": [],  # X component of orientation (Quaternion)
                        "xori_y": [],  # Y component of orientation (Quaternion)
                        "xori_z": [],  # Z component of orientation (Quaternion)
                        "xvel_x": [],  # X component of linear velocity
                        "xvel_y": [],  # Y component of linear velocity
                        "xvel_z": [],  # Z component of linear velocity
                        "xacc_x": [],  # X component of linear acceleration
                        "xacc_y": [],  # Y component of linear acceleration
                        "xacc_z": [],  # Z component of linear acceleration
                        "avel_x": [],  # X component of angular velocity
                        "avel_y": [],  # Y component of angular velocity
                        "avel_z": [],  # Z component of angular velocity
                        "traj_x": [],
                        "traj_y": [],
                        "lap_end":[]
                        }   
        
    def __len__(self):
        return len(self.data_logs["xpos_x"])