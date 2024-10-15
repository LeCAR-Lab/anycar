import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax import grad, jit, vmap
from functools import partial

import torch
from torch.utils.data import DataLoader, Dataset

import os
import glob
import time
import tqdm
import matplotlib.pyplot as plt
import pickle
import concurrent.futures
from transforms3d.euler import quat2euler

from flax import linen as nn

from car_foundation.utils import quaternion_to_euler, generate_subsequences, generate_subsequences_hf, align_yaw

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DynamicsDataset(Dataset):
    def __init__(self, path, sequence_length, mean=None, std=None):
        # self.data = np.loadtxt(path, delimiter=',')
        # self.episode_length = np.where(self.data[:, -1] == 1)[0][0] + 1
        # episode_terminations = np.arange(self.episode_length, self.data.shape[0], self.episode_length)
        # assert np.all(self.data[episode_terminations, -1] == 1), 'Episode terminations are not correct'
        # # createa third axis for the data according to the episode length
        # self.data = self.data.reshape(-1, self.episode_length, self.data.shape[1])
        # self.sequence_length = sequence_length
        # self.num_episodes = self.data.shape[0]
        # self.len = self.num_episodes * (self.episode_length - self.sequence_length)

        # find all csv files in the directory
        self.data = []
        for file in glob.glob(os.path.join(path, '*.csv')):
            self.data.append(np.loadtxt(file, delimiter=',').astype(np.float32))
        self.data = np.concatenate(self.data)
        self.episode_length = np.where(self.data[:, -1] == 1)[0][0] + 1
        episode_terminations = np.arange(self.episode_length - 1, self.data.shape[0], self.episode_length)
        assert np.all(self.data[episode_terminations, -1] == 1), 'Episode terminations are not correct'
        self.data[episode_terminations, -1] = np.arange(len(episode_terminations))
        # createa third axis for the data according to the episode length
        self.data = self.data.reshape(-1, self.episode_length, self.data.shape[1])
        self.sequence_length = sequence_length
        self.num_episodes = self.data.shape[0]
        self.len = self.num_episodes * (self.episode_length + 1 - self.sequence_length)

        if mean is not None:
            self.mean = mean
        else:
            self.mean = np.mean(self.data[:, :-1], axis=(0, 1))
        if std is not None:
            self.std = std
        else:
            self.std = np.std(self.data[:, :-1], axis=(0, 1))

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        episode_idx = idx // (self.episode_length + 1 - self.sequence_length)
        start_idx = idx % (self.episode_length + 1 - self.sequence_length)
        end_idx = start_idx + self.sequence_length
        return self.data[episode_idx, start_idx:end_idx]
    
    def get_episode(self, idx):
        return self.data[idx]
    
    # def parse_parameters(self, file_path):
        

class IssacSimDataset(Dataset):
    def __init__(self, path, sequence_length):
        data = np.load(path)
        obs = data['obs']
        action = data['action']
        reset = data['reset']
        self.data = np.concatenate([obs, action, reset[:, :, None]], axis=2)
        self.episode_length = self.data.shape[1]
        self.sequence_length = sequence_length
        self.num_episodes = self.data.shape[0]
        self.len = self.num_episodes * (self.episode_length + 1 - self.sequence_length)

        PX = 0
        PY = 1
        PZ = 2
        QW = 3
        QX = 4
        QY = 5
        QZ = 6
        VX = 7
        VY = 8
        VZ = 9
        WX = 10
        WY = 11
        WZ = 12
        UA = 13
        US = 14

        q = self.data[:, :, [QW, QX, QY, QZ]].reshape(-1, 4)
        _, _, y = quaternion_to_euler(q)
        self.data = self.data[:, :, [PX, PY, QW, VX, VY, WZ, UA, US, -1]]
        self.data[:, :, 2] = y.reshape(-1, self.episode_length)
        self.data = self.data.astype(np.float32)


    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        episode_idx = idx // (self.episode_length + 1 - self.sequence_length)
        start_idx = idx % (self.episode_length + 1 - self.sequence_length)
        end_idx = start_idx + self.sequence_length
        return self.data[episode_idx, start_idx:end_idx]
    
    def get_episode(self, idx):
        return self.data[idx]

class MujocoRawDataset:
    def __init__(self) -> None:
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
                        "traj_y": []
                        }
        self.car_params = {"mass": []}
        
    def generate_new_mass(self):
        default_mass = 3.794137
        lowermass= default_mass * 0.8
        uppermass = default_mass * 1.2
        self.car_params["mass"] = np.random.uniform(lowermass, uppermass)
        return self.car_params["mass"]
    
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
                        "traj_y": []
                        }
        
class MujocoDataset(Dataset):
    def __init__(self, path, history_length, action_length, 
                 delays=None, 
                 mean=None, 
                 std=None, 
                 teacher_forcing=True, 
                 binary_mask=False, 
                 use_jax=False,
                 attack=False,
                 filter=False,
                 add_noise=False,
    ):
        self.attack = attack
        self.add_noise = add_noise
        if delays is not None and any([d < 0 for d in delays]):
            raise ValueError('Delay should be greater than or equal to 0')

        
        def load_pickle(file):
            try:
                mujoco_raw_dataset = pickle.load(open(file, 'rb'))
            except:
                raise ValueError(f'Error loading the pickle file: {file}')
            q = np.array([mujoco_raw_dataset.data_logs["xori_w"],
                        mujoco_raw_dataset.data_logs["xori_x"],
                        mujoco_raw_dataset.data_logs["xori_y"],
                        mujoco_raw_dataset.data_logs["xori_z"]]).T
            _, _, y = quaternion_to_euler(q)
            data_array = np.array(
                [mujoco_raw_dataset.data_logs["xpos_x"],
                mujoco_raw_dataset.data_logs["xpos_y"],
                y,
                mujoco_raw_dataset.data_logs["xvel_x"],
                mujoco_raw_dataset.data_logs["xvel_y"],
                mujoco_raw_dataset.data_logs["avel_z"],
                mujoco_raw_dataset.data_logs["throttle"],
                mujoco_raw_dataset.data_logs["steer"],
                np.zeros_like(mujoco_raw_dataset.data_logs["xpos_x"]),
                ]).T
            episode_length = np.where(mujoco_raw_dataset.data_logs["lap_end"] == 1)[0][0] + 1
            # print("EPISODE LENGTH", episode_length)
            episode_terminations = np.arange(episode_length - 1, data_array.shape[0], episode_length)
            assert np.all(mujoco_raw_dataset.data_logs["lap_end"][episode_terminations] == 1), 'Episode terminations are not correct'
            data_array = data_array.reshape(-1, episode_length, data_array.shape[1])
            # shift the data by the delay
            if delays:
                data_array_delayed = []
                max_delay = max(delays)
                for delay in delays:
                    if delay == 0:
                        data_array_delayed.append(data_array[:, max_delay:, :])
                    else:
                        data_array_delayed.append(
                            np.concatenate([
                                data_array[:, :-delay, :6],
                                data_array[:, delay:, 6:],
                            ], axis=2)[:, max_delay-delay:, :]
                        )
                data_array = np.concatenate(data_array_delayed, axis=0)

            # remove the last few steps to make the data divisible by the sequence length
            # then reshape the data to have the sequence length as the third dimension
            episode_length = data_array.shape[1]
            data_array = data_array[:, :(episode_length - episode_length % self.sequence_length), :].reshape(-1, self.sequence_length, data_array.shape[2])

            return torch.tensor(data_array)

        if type(path) == str:
            pickle_files = glob.glob(os.path.join(path, '*.pkl'))
        elif type(path) == list:
            pickle_files = path
        else:
            raise ValueError('Path should be a string or a list of strings')
        if len(pickle_files) == 0:
            raise ValueError(f'No pickle files found in the directory: {path}')
        print(f'Loading {len(pickle_files)} pickle files')

        self.sequence_length = history_length + action_length
        
        # DEBUG = True
        DEBUG = False
        
        if DEBUG:
            self.data = []
            for pickle_file in pickle_files:
                self.data.append(load_pickle(pickle_file))
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                self.data = list(tqdm.tqdm(executor.map(load_pickle, pickle_files), total=len(pickle_files)))

        # concatenate all the episodes
        self.data = torch.concatenate(self.data, axis=0)
        self.data = self.data.to(torch.float32)

        #filter out high throttle data
        if filter:
            ##
            
            filtered_idx = []
            print("prefilted data shape:", self.data.shape)
            for i in range(self.data.shape[0]):
                # check if the car is on a straight away
                throttle = self.data[i, :, -3].numpy()
                vx = self.data[i, :, 3].numpy()
                vy = self.data[i, :, 4].numpy()

                if not (abs(vy/vx).max() > 1/3 and abs(vx).min() > 0.5):
                    filtered_idx.append(i)
                # if np.mean(throttle) < 0.8:
                    # filtered_idx.append(i)

            self.data = self.data[filtered_idx, :, :]
            print("filtered data shape:", self.data.shape)

        self.len = self.data.shape[0]

        # create a delta dataset
        self.delta_data = self.data.clone().detach()
        self.delta_data[:, 1:, :3] = self.data[:, 1:, :3] - self.data[:, :-1, :3]
        self.delta_data[:, 1:, 2] = align_yaw(self.delta_data[:, 1:, 2], 0.0)
        original_yaw = self.data[:, :-1, 2]
        # transform to Body Frame
        delta_x = self.delta_data[:, 1:, 0] * torch.cos(original_yaw) + self.delta_data[:, 1:, 1] * torch.sin(original_yaw)
        delta_y = -self.delta_data[:, 1:, 0] * torch.sin(original_yaw) + self.delta_data[:, 1:, 1] * torch.cos(original_yaw)
        
        self.delta_data[:, 1:, 0] = delta_x
        self.delta_data[:, 1:, 1] = delta_y
        self.delta_data = self.delta_data.detach()

        # split the delta data into history, action and future
        self.history = self.delta_data[:, :history_length, :8]
        self.action = self.delta_data[:, history_length-1:history_length+action_length-1, 6:8]
        self.y = self.delta_data[:, history_length:history_length+action_length, :6]

        # get the mean and std of the data
        if mean is not None:
            self.mean = mean
        else:
            self.mean = torch.mean(self.delta_data[:, 1:, :6], axis=(0, 1))
        if std is not None:
            self.std = std
        else:
            self.std = torch.std(self.delta_data[:, 1:, :6], axis=(0, 1))

        
        if self.attack:
            # add random noise to history
            noise_locations = torch.randint(0, history_length, (len(self.history),))
            self.history[range(len(self.history)), noise_locations, 3] += torch.rand(len(self.history)) * 60 - 30
            
        if self.add_noise:
            self.history[:, :, 0] += torch.rand_like(self.history[:, :, 0]) * 0.01 - 0.005
            self.history[:, :, 1] += torch.rand_like(self.history[:, :, 1]) * 0.01 - 0.005
            self.history[:, :, 2] += torch.rand_like(self.history[:, :, 2]) * 0.01 - 0.005
            self.history[:, :, 3] += torch.rand_like(self.history[:, :, 3]) * 1. - 0.5
            self.history[:, :, 4] += torch.rand_like(self.history[:, :, 4]) * 0.1 - 0.05
            self.history[:, :, 5] += torch.rand_like(self.history[:, :, 5]) * 0.01 - 0.005
            
            ## actions
            self.history[:, :, 6] += torch.rand_like(self.history[:, :, 6]) * 0.05 - 0.025
            self.history[:, :, 7] += torch.rand_like(self.history[:, :, 7]) * 0.05 - 0.025
        


        # expand the dataset for teacher forcing
        if teacher_forcing:
            if binary_mask:
                self.action, self.action_padding_mask = generate_subsequences_hf(self.action)
            else:
                self.action, self.action_padding_mask = generate_subsequences(self.action)
            self.history = torch.repeat_interleave(self.history, self.action.shape[1], dim=0)
            self.y = torch.repeat_interleave(self.y, self.action.shape[1], dim=0)
            self.len = self.history.shape[0]
            # self.data = torch.cat([self.history, 
            #                        torch.cat(
            #                            [self.y, torch.cat(
            #                                [self.action[:, 1:, :], torch.zeros_like(self.action[:, 0:1, :])], axis=1
            #                            )], axis=2)
            #                       ], axis=1)
            self.data = torch.repeat_interleave(self.data, self.action.shape[1], dim=0)
        else:
            if binary_mask:
                self.action_padding_mask = torch.ones(self.action.shape[0], self.action.shape[1])
            else:
                self.action_padding_mask = torch.zeros(self.action.shape[0], self.action.shape[1])

        # convert to jax array if needed
        if use_jax:
            self.data = jax.device_put(self.data, jax.devices('cpu')[0])
            self.history = jax.device_put(self.history, jax.devices('cpu')[0])
            self.action = jax.device_put(self.action, jax.devices('cpu')[0])
            self.y = jax.device_put(self.y, jax.devices('cpu')[0])
            if binary_mask:
                self.action_padding_mask = jax.device_put(self.action_padding_mask, jax.devices('cpu')[0])
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        action_padding_mask = None if self.action_padding_mask is None else self.action_padding_mask[idx]
        return self.history[idx], self.action[idx], self.y[idx], action_padding_mask, self.data[idx]
    
    def get_episode(self, idx):
        return self.data[idx]
    