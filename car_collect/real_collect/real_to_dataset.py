from car_foundation import CAR_FOUNDATION_DATA_DIR
import numpy as np
import os
import pickle
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from car_dataset import CarDataset
import time
import datetime
from tqdm import tqdm
import math


import faulthandler
faulthandler.enable()

def log_data(dataset, raw_obs, raw_action):
        dataset.data_logs["xpos_x"].append(raw_obs[0])
        dataset.data_logs["xpos_y"].append(raw_obs[1])
        dataset.data_logs["xpos_z"].append(0)
        #log orientation
        r = 0
        p = 0
        y = raw_obs[2]
        r = R.from_euler("xyz", np.array([r, p, y]))
        quat = r.as_quat()
        dataset.data_logs["xori_w"].append(quat[3])
        dataset.data_logs["xori_x"].append(quat[0])
        dataset.data_logs["xori_y"].append(quat[1])
        dataset.data_logs["xori_z"].append(quat[2])
        #log linear velocity
        dataset.data_logs["xvel_x"].append(raw_obs[3])
        dataset.data_logs["xvel_y"].append(raw_obs[4])
        dataset.data_logs["xvel_z"].append(0)
        #log linear acceleration
        dataset.data_logs["xacc_x"].append(0)
        dataset.data_logs["xacc_y"].append(0)
        dataset.data_logs["xacc_z"].append(0)
        #log angular velocity
        dataset.data_logs["avel_x"].append(0)
        dataset.data_logs["avel_y"].append(0)
        dataset.data_logs["avel_z"].append(raw_obs[5])

        dataset.data_logs["traj_x"].append(0)
        dataset.data_logs["traj_y"].append(0)

        dataset.data_logs["lap_end"].append(0)

        dataset.data_logs["throttle"].append(raw_action[0])
        dataset.data_logs["steer"].append(raw_action[1])

# @ray.remote
def rollout(id, debug_plots, datadir, raw_obs, raw_action):
    tic = time.time()

    dataset = CarDataset()
    dataset.car_params['sim'] = "real"
    
    # fine tuning
    dataset.car_params['mass'] = None
    dataset.car_params['friction'] = None
    dataset.car_params['max_throttle'] = None
    dataset.car_params['delay'] = 0

    print("Car Params", dataset.car_params)

    dataset.car_params["wheelbase"] = 0.32 #32 centimeters 0)

    total_timesteps = raw_obs.shape[0]
    
    for t in tqdm(range(total_timesteps)):
        log_data(dataset, raw_obs[t,:], raw_action[t,:])
        
    dataset.data_logs["lap_end"][-1] = 1 
    now = datetime.datetime.now().isoformat(timespec='milliseconds')
    file_name = "log_" + str(id) + '_' + str(now) + ".pkl"
    filepath = os.path.join(datadir, file_name)
    
    for key, value in dataset.data_logs.items():
        dataset.data_logs[key] = np.array(value)

    with open(filepath, 'wb') as outp: 
        pickle.dump(dataset, outp, pickle.HIGHEST_PROTOCOL)

    print("Saved Data to:", filepath)

    if debug_plots:
        plt.plot(dataset.data_logs["xpos_x"], dataset.data_logs["xpos_y"], linestyle = "dashed", label='Car Position')
        plt.show()
        
        dataset.reset_logs()

    print("Data Conversion Complete!")
    print("Total Timesteps: ", total_timesteps)
    print("Elapsed_Time: ", time.time() - tic)
    return True

if __name__ == "__main__":

    debug_plots = False
    data_dir = os.path.join(CAR_FOUNDATION_DATA_DIR, "real")
    os.makedirs(data_dir, exist_ok=True)
    num_success = 0
    filepath = "dataset.pkl"
    if os.path.getsize(filepath) == 0:
        print("File is empty.")
    else:
        print("File size is:", os.path.getsize(filepath))
    with open(filepath, 'rb') as outp: 
        dataset = pickle.load(outp)
    print(np.array(dataset["obs_list"]).shape)
    print(np.array(dataset["action_list"]).shape)
    dataset = [(np.array(dataset["obs_list"]), np.array(dataset["action_list"]))]
    # x, y, yaw, vx, vy, w
    #augment data
    og_throttle = dataset[0][1][:,0]
    augmented_steer = -1*dataset[0][1][:,1] #flip the driving
    og_x = dataset[0][0][:, 0]
    og_y = dataset[0][0][:, 1]
    og_yaw = dataset[0][0][:, 2]
    og_vx = dataset[0][0][:, 3]
    augmented_vy = -1*dataset[0][0][:, 4] #flip
    augmented_w = -1*dataset[0][0][:, 5] #flip

    new_x = [og_x[0]]
    new_y = [og_y[0]]
    new_yaw = [og_yaw[0]]

    prev_frame = np.array([[math.cos(og_yaw[0]), -math.sin(og_yaw[0]), og_x[0]],
                  [math.sin(og_yaw[0]), math.cos(og_yaw[0]), og_y[0]],
                  [0, 0 , 1]])
    
    flipped_frame = np.array([[math.cos(og_yaw[0]), -math.sin(og_yaw[0]), og_x[0]],
                  [math.sin(og_yaw[0]), math.cos(og_yaw[0]), og_y[0]],
                  [0, 0 , 1]])

    for i in range(1, len(dataset[0][0])):

        delta_frame = np.linalg.inv(prev_frame) @ np.array([[math.cos(og_yaw[i]), -math.sin(og_yaw[i]), og_x[i]],
                                                        [math.sin(og_yaw[i]), math.cos(og_yaw[i]), og_y[i]],
                                                        [0, 0 , 1]])
        
        prev_frame = np.array([[math.cos(og_yaw[i]), -math.sin(og_yaw[i]), og_x[i]],
                                [math.sin(og_yaw[i]), math.cos(og_yaw[i]), og_y[i]],
                                [0, 0 , 1]])

        dx = delta_frame[0][2]
        dy = delta_frame[1][2]
        dyaw = math.atan2(delta_frame[1][0], delta_frame[0][0])

        # input()
        augmented_dy = dy * -1
        augmented_dyaw = -1 * dyaw

        flipped_frame = flipped_frame @ np.array([[math.cos(augmented_dyaw), -math.sin(augmented_dyaw), dx],
                                                        [math.sin(augmented_dyaw), math.cos(augmented_dyaw), augmented_dy],
                                                        [0, 0 , 1]])

        new_x.append(flipped_frame[0][2])
        new_y.append(flipped_frame[1][2])
        new_yaw.append(math.atan2(flipped_frame[1][0], flipped_frame[0][0]))

    new_x = np.array(new_x)
    new_y = np.array(new_y)
    new_yaw = np.array(new_yaw)
    
    flipped_obs_list = np.array([new_x, new_y, new_yaw, og_vx, augmented_vy, augmented_w]).transpose()
    flipped_action_list = np.array([og_throttle, augmented_steer]).transpose()

    print(flipped_obs_list.shape)
    print(flipped_action_list.shape)

    dataset.append((flipped_obs_list, flipped_action_list))

    for i in range(len(dataset)):
        raw_obs = dataset[i][0]
        raw_action = dataset[i][1]
        ret = rollout(i, debug_plots, data_dir, raw_obs, raw_action)
        print(f"Episode {i} Complete")
        if ret:
            num_success += 1

        
