from copy import deepcopy
import os
import json
import torch
import scipy
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from car_dynamics.analysis import pos2vel_savgol, calc_delta_v


class DynDataset(Dataset):
    def __init__(self, inputs, labels):

        # self.input_dims = input_dims
        # self.output_dims = output_dims
        assert inputs.shape[0] == labels.shape[0]
        self.data = inputs
        self.labels = labels
        self.length = inputs.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def append(self, data_point, label):
        '''Assume input is numpy.array'''
        assert data_point.shape[0] == self.input_dims
        assert label.shape[0] == self.output_dims
        self.data[self.length] = data_point
        self.labels[self.length] = label
        self.length += 1
        
    def get_subset(self, indices):
        dataset = DynDataset(deepcopy(self.data[indices]), deepcopy(self.labels[indices]))
        return dataset

    def to_tensor(self,device):
        self.data = torch.tensor(self.data, dtype=torch.float32, device=device)
        self.labels = torch.tensor(self.labels, dtype=torch.float32,device=device)

def normalize_dataset(dataset, method='minmax'):
    # import pdb; pdb.set_trace()
    if method == 'minmax':
        data_min = torch.min(dataset.data[:dataset.length], dim=0, keepdim=True).values
        data_max = torch.max(dataset.data[:dataset.length], dim=0, keepdim=True).values
        labels_min = torch.min(dataset.labels[:dataset.length], dim=0, keepdim=True).values
        labels_max = torch.max(dataset.labels[:dataset.length], dim=0, keepdim=True).values

        dataset.data[:dataset.length] = (dataset.data[:dataset.length] - data_min) / \
                                                                (data_max - data_min + 1e-8)
        dataset.labels[:dataset.length] = (dataset.labels[:dataset.length] - labels_min) / \
                                                                (labels_max - labels_min + 1e-8)

        stats = {
            'normalization': 'minmax',
            'input_min': data_min,
            'input_max': data_max,
            'output_min': labels_min,
            'output_max': labels_max,
        }
    elif method == 'z-score':
        data_mean = torch.mean(dataset.data[:dataset.length], dim=0, keepdim=True)
        data_std = torch.std(dataset.data[:dataset.length], dim=0, keepdim=True)
        labels_mean = torch.mean(dataset.labels[:dataset.length], dim=0, keepdim=True)
        labels_std = torch.std(dataset.labels[:dataset.length], dim=0, keepdim=True)

        dataset.data[:dataset.length] = (dataset.data[:dataset.length] - data_mean) / \
                                                                (data_std + 1e-8)
        dataset.labels[:dataset.length] = (dataset.labels[:dataset.length] - labels_mean) / \
                                                                (labels_std + 1e-8)

        stats = {
            'normalization': 'z-score',
            'input_mean': data_mean,
            'input_std': data_std,
            'output_mean': labels_mean,
            'output_std': labels_std,
        }

    return stats

def clean_random_data(obs_list, action_list, is_recover, H):
    """
    Return:
        - state_all: list, each element is a tuple (traj, actions)
    """
    if len(is_recover) == 0:
        print("Warning: is_recover is None, use all data")
        is_recover = np.zeros(len(obs_list), dtype=bool)
    state_all = []
    obs_holder = []
    action_holder = []
    in_sequence = False
    for i, obs in enumerate(obs_list):
        if is_recover[i] == False:
            obs_holder.append(obs)
            action_holder.append(action_list[i])
            in_sequence = True
        elif in_sequence:
            traj = np.array(obs_holder)
            actions = np.array(action_holder)
            state_all.append((traj, actions))
            obs_holder = []
            action_holder = []
            in_sequence = False
    if in_sequence:
        traj = np.array(obs_holder)
        actions = np.array(action_holder)
        state_all.append((traj, actions))
        
    state_clean = []
    for state in state_all:
        if state[0].shape[0] >= H:
            state_clean.append(state)
            
    return state_clean
    

def parse_data(H, data_list, load_state, proj_dir, log_dir, LF, LR):
    # dt = 0.125 
    x_list = []
    y_list = []
    for data_point in data_list:
        with open(os.path.join(proj_dir, log_dir, data_point['dir'], 'header.json')) as f:
            header_info = json.load(f)
        t_list, p_dict, yaw_dict, action_list, controller_info = load_state(
            os.path.join(proj_dir, log_dir, data_point['dir']), data_point['range'], orientation_provider="ORIENTATION_PROVIDOER")
        if data_point['name'] == 'real-random':
            state_all = clean_random_data(p_dict['obs'], action_list, controller_info['is_recover'], H)
        else:
            state_all = [(np.array(p_dict['obs']), action_list)]
        
        for obs_np, actions in state_all:
            # obs_np = p_dict['obs']
            assert np.all(obs_np[:, 3] >= 0.)
            # p_smooth, vel, vel_vec = pos2vel_savgol(obs_np[:,:2],window_length=5, delta=dt)
            # delta, beta = calc_delta_v(vel_vec, obs_np[:,2], LF, LR)
            vel_vec = obs_np[1:, :2] - obs_np[:-1, :2]
            delta, beta = calc_delta_v(vel_vec, obs_np[:,2], LF, LR)
            # import pdb; pdb.set_trace()
            x_train_np = np.concatenate((obs_np[:, 0:4], actions[:, :2]), axis=1)
            # print(x_train_np.shape, y_train_np.shape)
            ## Stackup history
            x_train_hist_list = []
            for i in range(H-1, x_train_np.shape[0]-1):
                x_datapoint = x_train_np[i+1-H:i+1].copy()
                x_datapoint[:, :2] -= x_datapoint[0, :2]
                x_train_hist_list.append(x_datapoint.flatten())

            y_train_np = beta[H-1:]
            vel_inc_np = obs_np[H:, 3] - obs_np[H-1:-1, 3]

            x_list += x_train_hist_list
            y_list += np.column_stack((np.cos(y_train_np), np.sin(y_train_np), vel_inc_np)).tolist()
        
    x_list = np.array(x_list)
    y_list = np.array(y_list)

    X_train = X_train = torch.Tensor(x_list)
    y_train = torch.Tensor(y_list)
    return X_train, y_train

def parse_data_end2end(H, data_list, load_state, proj_dir, log_dir,):
    # dt = 0.125 
    x_list = []
    y_list = []
    for data_point in data_list:
        with open(os.path.join(proj_dir, log_dir, data_point['dir'], 'header.json')) as f:
            header_info = json.load(f)
        t_list, p_dict, yaw_dict, action_list, controller_info = load_state(
            os.path.join(proj_dir, log_dir, data_point['dir']), data_point['range'], orientation_provider="ORIENTATION_PROVIDOER")
        is_recover = controller_info['is_recover']
        obs_np = p_dict['obs']
        # is_recover[obs_np[:,3] < 0.3] = True
        
        if data_point['name'] == 'real-random':
            state_all = clean_random_data(obs_np, action_list, is_recover, H)
        else:
            state_all = [(np.array(p_dict['obs']), action_list)]
        
        for obs_np, actions in state_all:
            
            # obs_np = p_dict['obs']
            
            # obs_norm = np.zeros((obs_np.shape[0], 5))
            # obs_norm[:, :2] = obs_np[:, :2]
            # obs_norm[:, 2] = np.cos(obs_np[:, 2])
            # obs_norm[:, 3] = np.sin(obs_np[:, 2])
            # obs_norm[:, 4] = obs_np[:, 3]
            obs_norm = obs_np[:, :4] + 0.
            
            x_train_np = np.concatenate((obs_norm, actions[:, :2]), axis=1)

            ## Stackup history
            x_train_hist_list = []
            for i in range(H-1, x_train_np.shape[0]-1):
                x_datapoint = x_train_np[i+1-H:i+1].copy()
                x_datapoint[:, :2] -= x_datapoint[0, :2]
                x_train_hist_list.append(x_datapoint.flatten())

            y_train_np = obs_norm[H:] - obs_norm[H-1:-1]
            # y_train_np[:, 2] = np.cos(obs_np[H:, 2] - obs_np[H-1:-1, 2])
            # y_train_np[:, 3] = np.sin(obs_np[H:, 2] - obs_np[H-1:-1, 2])

            x_list += x_train_hist_list
            y_list += y_train_np.tolist()
        
    x_list = np.array(x_list)
    # import pdb; pdb.set_trace()
    # assert np.all(x_list[:, 4] >= 0.)
    assert np.all(x_list[:, 3] >= 0.)
    y_list = np.array(y_list)

    X_train = X_train = torch.Tensor(x_list)
    y_train = torch.Tensor(y_list)
    return X_train, y_train

def parse_data_heading_psi(H, data_list, load_state, proj_dir, log_dir, LF, LR):
    # dt = 0.125 
    x_list = []
    y_list = []
    for data_point in data_list:
        with open(os.path.join(proj_dir, log_dir, data_point['dir'], 'header.json')) as f:
            header_info = json.load(f)
        t_list, p_dict, yaw_dict, action_list, controller_info = load_state(
            os.path.join(proj_dir, log_dir, data_point['dir']), data_point['range'], orientation_provider="ORIENTATION_PROVIDOER")
        if data_point['name'] == 'real-random':
            state_all = clean_random_data(p_dict['obs'], action_list, controller_info['is_recover'], H)
        else:
            state_all = [(np.array(p_dict['obs']), action_list)]
        
        for obs_np, actions in state_all:
            # obs_np = p_dict['obs']
            assert np.all(obs_np[:, 3] >= 0.)
            # p_smooth, vel, vel_vec = pos2vel_savgol(obs_np[:,:2],window_length=5, delta=dt)
            # delta, beta = calc_delta_v(vel_vec, obs_np[:,2], LF, LR)
            vel_vec = obs_np[1:, :2] - obs_np[:-1, :2]
            delta, beta = calc_delta_v(vel_vec, obs_np[:,2], LF, LR)
            # import pdb; pdb.set_trace()
            x_train_np = np.concatenate((obs_np[:, 0:4], actions[:, :2]), axis=1)
            # print(x_train_np.shape, y_train_np.shape)
            ## Stackup history
            x_train_hist_list = []
            for i in range(H-1, x_train_np.shape[0]-1):
                x_datapoint = x_train_np[i+1-H:i+1].copy()
                x_datapoint[:, :2] -= x_datapoint[0, :2]
                x_train_hist_list.append(x_datapoint.flatten())

            y_train_np = beta[H-1:]
            psi_next = obs_np[H:, 2] - obs_np[H-1:-1, 2]
            vel_inc_np = obs_np[H:, 3] - obs_np[H-1:-1, 3]

            x_list += x_train_hist_list
            y_list += np.column_stack((np.cos(y_train_np), np.sin(y_train_np), 
                                    np.cos(psi_next), np.sin(psi_next), 
                                    vel_inc_np)).tolist()
        
    x_list = np.array(x_list)
    y_list = np.array(y_list)

    X_train = X_train = torch.Tensor(x_list)
    y_train = torch.Tensor(y_list)
    return X_train, y_train


def parse_data_end2end_norm(H, data_list, load_state, proj_dir, log_dir, dt, smooth_action=True, smooth_vel=False, smooth_yaw=False):
    # dt = 0.125 
    x_list = []
    y_list = []
    for data_point in data_list:
        with open(os.path.join(proj_dir, log_dir, data_point['dir'], 'header.json')) as f:
            header_info = json.load(f)
        t_list, p_dict, yaw_dict, action_list_raw, controller_info = load_state(
            os.path.join(proj_dir, log_dir, data_point['dir']), data_point['range'], orientation_provider="ORIENTATION_PROVIDOER")
        is_recover = controller_info['is_recover']
        obs_extract = p_dict['obs']
        
        ## smooth action
        if smooth_action:
            action_list, _, _ = pos2vel_savgol(action_list_raw,window_length=10, delta=dt)
        else:
            action_list = action_list_raw + 0.
        
        if smooth_yaw:
            obs_extract[:, 2] = scipy.signal.savgol_filter(obs_extract[:, 2], window_length=5, polyorder=3, deriv=0, delta=dt, )
        
        if 'limit' in data_point.keys():
            obs_extract = obs_extract[:data_point['limit']]
            action_list = action_list[:data_point['limit']]
            if len(is_recover) > 0:
                is_recover = is_recover[:data_point['limit']]
            
        if data_point['name'] == 'real-random':
            is_recover[obs_extract[:,3] < 0.1] = True
            state_all = clean_random_data(obs_extract, action_list, is_recover, H)
        else:
            state_all = [(obs_extract, action_list)]
        
        for obs_np, actions in state_all:
            if len(obs_np) < 40:
                continue
            # obs_np = p_dict['obs']
            # p_smooth, vel, vel_vec = pos2vel_savgol(obs_np[:,:2],window_length=5, delta=dt)
            # obs_np[:, 3] = vel
            if not smooth_vel:
                vel = np.linalg.norm(obs_np[1:, :2] - obs_np[:-1, :2], axis=1) / dt
            else:
                _, vel, _ = pos2vel_savgol(obs_np[:,:2],window_length=5, delta=dt)
                vel = vel[:-1]
            obs_np = obs_np[:-1]
            actions = actions[:-1]
            obs_np[:, 3] = vel
            
            obs_norm = np.zeros((obs_np.shape[0], 5))
            obs_norm[:, :2] = obs_np[:, :2]
            obs_norm[:, 2] = np.cos(obs_np[:, 2])
            obs_norm[:, 3] = np.sin(obs_np[:, 2])
            obs_norm[:, 4] = obs_np[:, 3]
            
            x_train_np = np.concatenate((obs_norm, actions[:, :2]), axis=1)

            ## Stackup history
            x_train_hist_list = []
            for i in range(H-1, x_train_np.shape[0]-1):
                x_datapoint = x_train_np[i+1-H:i+1].copy()
                x_datapoint[:, :2] -= x_datapoint[0, :2]
                x_train_hist_list.append(x_datapoint.flatten())

            y_train_np = (obs_norm[H:] - obs_norm[H-1:-1]) / dt
            
            # y_train_np[:, 2] = np.cos((obs_np[H:, 2] - obs_np[H-1:-1, 2])/dt)
            # y_train_np[:, 3] = np.sin((obs_np[H:, 2] - obs_np[H-1:-1, 2])/dt)
            # y_train_np = (obs_np[H:] - obs_np[H-1:-1]) / dt
            # y_train_np[:, 2] = y_train_np[:, 3]
            # y_train_np = y_train_np[:, :3]
            # dpsi = obs_np[H:, 2] - obs_np[H-1:-1, 2]
            # y_train_np[:, 2] = np.cos(dpsi)
            # y_train_np[:, 3] = np.sin(dpsi)

            ## temporarily remove dvel
            # y_train_np = y_train_np[:, :4]
            
            # ## temporarily only keep dvel
            # y_train_np = y_train_np[:, 4:5]
            
            x_list += x_train_hist_list
            y_list += y_train_np.tolist()
        
    x_list = np.array(x_list)
    # import pdb; pdb.set_trace()
    assert np.all(x_list[:, 4] >= 0.)
    # assert np.all(x_list[:, 3] >= 0.)
    y_list = np.array(y_list)

    # remove action
    # x_list = x_list[:, :-2]
    
    X_train = X_train = torch.Tensor(x_list)
    y_train = torch.Tensor(y_list)
    return X_train, y_train


def parse_data_end2end_extend(H, data_list, load_state, proj_dir, log_dir, dt, smooth_action=True, smooth_vel=False, smooth_yaw=False):
    # dt = 0.125 
    x_list = []
    y_list = []
    for data_point in data_list:
        with open(os.path.join(proj_dir, log_dir, data_point['dir'], 'header.json')) as f:
            header_info = json.load(f)
        t_list, p_dict, yaw_dict, action_list_raw, controller_info = load_state(
            os.path.join(proj_dir, log_dir, data_point['dir']), data_point['range'], orientation_provider="ORIENTATION_PROVIDOER")
        is_recover = controller_info['is_recover']
        obs_extract = p_dict['obs']
        
        ## smooth action
        if smooth_action:
            action_list, _, _ = pos2vel_savgol(action_list_raw,window_length=10, delta=dt)
        else:
            action_list = action_list_raw + 0.
        
        if smooth_yaw:
            obs_extract[:, 2] = scipy.signal.savgol_filter(obs_extract[:, 2], window_length=5, polyorder=3, deriv=0, delta=dt, )
        
        if 'limit' in data_point.keys():
            obs_extract = obs_extract[:data_point['limit']]
            action_list = action_list[:data_point['limit']]
            if len(is_recover) > 0:
                is_recover = is_recover[:data_point['limit']]
            
        if data_point['name'] == 'real-random':
            is_recover[obs_extract[:,3] < 0.1] = True
            state_all = clean_random_data(obs_extract, action_list, is_recover, H)
        else:
            state_all = [(obs_extract, action_list)]
        
        for obs_np, actions in state_all:
            if len(obs_np) < 40:
                continue
            # obs_np = p_dict['obs']
            # p_smooth, vel, vel_vec = pos2vel_savgol(obs_np[:,:2],window_length=5, delta=dt)
            # obs_np[:, 3] = vel
            if not smooth_vel:
                vel = np.linalg.norm(obs_np[1:, :2] - obs_np[:-1, :2], axis=1) / dt
            else:
                _, vel, _ = pos2vel_savgol(obs_np[:,:2],window_length=5, delta=dt)
                vel = vel[:-1]
                
            ## Calc beta and omega
            # diff_pos = obs_np[1:, :2] - obs_np[:-1, :2]
            # beta = np.arctan2(diff_pos[:, 1], diff_pos[:, 0]) - obs_np[:-1, 2]
            # beta = np.arctan2(np.sin(beta), np.cos(beta))
            # diff_yaw = obs_np[1:, 2] - obs_np[:-1, 2]
            # omega = np.arctan2(np.sin(diff_yaw), np.cos(diff_yaw)) / dt
            
            # obs_np = obs_np[:-1]
            # actions = actions[:-1]
            # obs_np[:, 3] = vel
            
            # obs_extend = np.zeros((obs_np.shape[0], 6)) # [x,y,theta,vx,vy,omega]
            # obs_extend[:, :2] = obs_np[:, :2]
            # obs_extend[:, 2] = obs_np[:, 2]
            # obs_extend[:, 3] = np.cos(beta) * obs_np[:, 3]
            # obs_extend[:, 4] = np.sin(beta) * obs_np[:, 3]
            # obs_extend[:, 5] = omega
            obs_extend = deepcopy(obs_np)
            assert obs_extend.shape[1] == 6, f"obs_extend.shape[0] = {obs_extend.shape[1]}"
            x_train_np = np.concatenate((obs_extend, actions[:, :2]), axis=1)

            ## Stackup history
            x_train_hist_list = []
            for i in range(H-1, x_train_np.shape[0]-1):
                x_datapoint = x_train_np[i+1-H:i+1].copy()
                x_datapoint[:, :2] -= x_datapoint[0, :2]
                x_train_hist_list.append(x_datapoint.flatten())

            
            y_train_np = (obs_extend[H:] - obs_extend[H-1:-1]) / dt
            
            # y_train_np[:, 2] = obs_extend[H:, 2] - obs_extend[H-1:-1, 2]
            # y_train_np[:, 2] = np.arctan2(np.sin(y_train_np[:, 2]), np.cos(y_train_np[:, 2])) / dt
            
            ## temporarily only keep dvx, dvy, domega
            # y_train_np = y_train_np[:, 3:]
            
            x_list += x_train_hist_list
            y_list += y_train_np.tolist()
        
    x_list = np.array(x_list)
    # import pdb; pdb.set_trace()
    # assert np.all(x_list[:, 4] >= 0.)
    # assert np.all(x_list[:, 3] >= 0.)
    y_list = np.array(y_list)

    # remove action
    # x_list = x_list[:, :-2]
    
    X_train = X_train = torch.Tensor(x_list)
    y_train = torch.Tensor(y_list)
    return X_train, y_train


def parse_data_end2end_8dim(H, data_list, log_dir, smooth_action=False, smooth_vel=False, smooth_yaw=False):
    # dt = 0.125 
    x_list = []
    y_list = []
    for data_point in data_list:
        df = pd.read_csv(os.path.join(log_dir, data_point['dir']))
        
        t = df['time'].to_numpy()
        dt = t[1:] - t[:-1]
        pos_x = df['pos_x'].to_numpy()
        pos_y = df['pos_y'].to_numpy()
        yaw = df['yaw'].to_numpy()
        target_vel = df['target_vel'].to_numpy()
        target_steer = df['target_steer'].to_numpy()
        
        print("DATASET MEAN DT: ", dt.mean())
        ## smooth action
        if smooth_action:
            target_vel = ...
            target_steer = ...
        
        if smooth_yaw:
            yaw = ...
        
        vel_vec = np.stack((pos_x[1:] - pos_x[:-1], pos_y[1:] - pos_y[:-1]), axis=1) / np.expand_dims(dt, axis=1)   
        vx = vel_vec[:, 0] * np.cos(yaw[:-1]) + vel_vec[:, 1] * np.sin(yaw[:-1])
        vy = vel_vec[:, 1] * np.cos(yaw[:-1]) - vel_vec[:, 0] * np.sin(yaw[:-1])
        omega = (yaw[1:] - yaw[:-1]) / dt
        
        x_train_np = np.stack((pos_x[:-1], pos_y[:-1], yaw[:-1], vx, vy, omega, 
                                target_vel[:-1], target_steer[:-1]), axis=1)

        ## Stackup history
        x_train_hist_list = []
        for i in range(H-1, x_train_np.shape[0]-1):
            x_datapoint = x_train_np[i+1-H:i+1].copy()
            # normalize x, y
            x_datapoint[:, :2] -= x_datapoint[0, :2]
            x_train_hist_list.append(x_datapoint.flatten())

        
        y_train_np = (x_train_np[H:, :-2] - x_train_np[H-1:-1, :-2]) / np.expand_dims(dt[H-1:-1], axis=1)
      
        x_list += x_train_hist_list
        y_list += y_train_np.tolist()
        
    x_list = np.array(x_list)

    y_list = np.array(y_list)

    X_train = X_train = torch.Tensor(x_list)
    y_train = torch.Tensor(y_list)
    print(X_train.shape, y_train.shape)
    return X_train, y_train