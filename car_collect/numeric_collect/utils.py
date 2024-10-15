import os
from car_dataset import CarDataset
from car_dynamics.envs.numeric_sim.car_numeric import Car2D

def mkdir_if_not_exist(path):
        if not os.path.exists(path):
                os.makedirs(path)

def log_data(dataset: CarDataset, env: Car2D, action, controller):
        dataset.data_logs["xpos_x"].append(env.car_pos[0])
        dataset.data_logs["xpos_y"].append(env.car_pos[1])
        dataset.data_logs["xpos_z"].append(env.car_pos[2])
        #log orientation
        dataset.data_logs["xori_w"].append(env.car_orientation[0])
        dataset.data_logs["xori_x"].append(env.car_orientation[1])
        dataset.data_logs["xori_y"].append(env.car_orientation[2])
        dataset.data_logs["xori_z"].append(env.car_orientation[3])
        #log linear velocity
        dataset.data_logs["xvel_x"].append(env.car_lin_vel[0])
        dataset.data_logs["xvel_y"].append(env.car_lin_vel[1])
        dataset.data_logs["xvel_z"].append(env.car_lin_vel[2])
        #log linear acceleration
        dataset.data_logs["xacc_x"].append(env.car_lin_acc[0])
        dataset.data_logs["xacc_y"].append(env.car_lin_acc[1])
        dataset.data_logs["xacc_z"].append(env.car_lin_acc[2])
        #log angular velocity
        dataset.data_logs["avel_x"].append(env.car_ang_vel[0])
        dataset.data_logs["avel_y"].append(env.car_ang_vel[1])
        dataset.data_logs["avel_z"].append(env.car_ang_vel[2])

        dataset.data_logs["traj_x"].append(controller.target_pos[0])
        dataset.data_logs["traj_y"].append(controller.target_pos[1])

        dataset.data_logs["lap_end"].append(0)

        dataset.data_logs["throttle"].append(action[0])
        dataset.data_logs["steer"].append(action[1])
        
def log_data_minimal(dataset: CarDataset, env: Car2D, action):
        dataset.data_logs["xpos_x"].append(env.car_pos[0])
        dataset.data_logs["xpos_y"].append(env.car_pos[1])
        dataset.data_logs["xpos_z"].append(env.car_pos[2])
        #log orientation
        dataset.data_logs["xori_w"].append(env.car_orientation[0])
        dataset.data_logs["xori_x"].append(env.car_orientation[1])
        dataset.data_logs["xori_y"].append(env.car_orientation[2])
        dataset.data_logs["xori_z"].append(env.car_orientation[3])
        #log linear velocity
        dataset.data_logs["xvel_x"].append(env.car_lin_vel[0])
        dataset.data_logs["xvel_y"].append(env.car_lin_vel[1])
        dataset.data_logs["xvel_z"].append(env.car_lin_vel[2])
        #log linear acceleration
        dataset.data_logs["xacc_x"].append(env.car_lin_acc[0])
        dataset.data_logs["xacc_y"].append(env.car_lin_acc[1])
        dataset.data_logs["xacc_z"].append(env.car_lin_acc[2])
        #log angular velocity
        dataset.data_logs["avel_x"].append(env.car_ang_vel[0])
        dataset.data_logs["avel_y"].append(env.car_ang_vel[1])
        dataset.data_logs["avel_z"].append(env.car_ang_vel[2])

        dataset.data_logs["lap_end"].append(0)
        dataset.data_logs["throttle"].append(action[0])
        dataset.data_logs["steer"].append(action[1])