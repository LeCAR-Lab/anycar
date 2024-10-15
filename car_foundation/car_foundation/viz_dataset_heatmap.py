import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from car_foundation import CAR_FOUNDATION_DATA_DIR
from car_foundation.dataset import MujocoDataset
from car_dataset import CarDataset

dataset_name = 'XXX'

dataset = MujocoDataset(os.path.join(CAR_FOUNDATION_DATA_DIR, dataset_name), 50, 50, teacher_forcing=False)
throttle = dataset.data[:, :, -3].numpy()
steer = dataset.data[:, :, -2].numpy()
yaw = dataset.data[:, :, 2].numpy()
vx = dataset.data[:, :, 3].numpy()
vy = dataset.data[:, :, 4].numpy()
vyaw = dataset.data[:, :, 5].numpy()
lin_vel_x = dataset.data[:, :, 3].numpy()
print(np.min(throttle), np.max(throttle))
print(np.min(steer), np.max(steer))
print(np.min(yaw), np.max(yaw))
print(np.min(vx), np.max(vx))
print(np.min(vy), np.max(vy))
print(np.min(vyaw), np.max(vyaw))

def filter(data, percentile=95):
    low = np.percentile(data, (100 - percentile) / 2)
    high = np.percentile(data, 100 - (100 - percentile) / 2)
    return np.clip(data, low, high)


# visualize the 2D heatmap of the throttle and steer
plt.figure()
plt.hist2d(throttle.flatten(), steer.flatten(), bins=100, cmap='viridis')
plt.xlim([-1., 1.])
plt.ylim([-1., 1.])
plt.colorbar()
plt.xlabel('Throttle')
plt.ylabel('Steer')
plt.title(f'{dataset_name} - Throttle and Steer Distribution')
# plt.savefig('throttle_steer_distribution.png')

# plt.figure()
# plt.hist(throttle.flatten(), bins=1000, color='red')
# plt.title('Throttle Distribution')



plt.figure()
plt.hist2d(filter(vx.flatten()), filter(vy.flatten()), bins=100, cmap='viridis')
plt.colorbar()
plt.xlabel('Vx')
plt.ylabel('Vy')
plt.title(f'{dataset_name} - Vx and Vy Distribution')
# plt.savefig('vx_vy_distribution.png')
# plt.savefig('throttle_steer_distribution.png')

# plt.show()

plt.figure()
plt.hist(lin_vel_x.flatten(), bins=100, color='red')
plt.xlabel('Linear Velocity x')
plt.ylabel('Frequency')
# plt.plot(lin_vel_x.flatten())
plt.title(f'{dataset_name} - Linear Velocity x Distribution')

# plt.show()


# plt.figure()

# files = os.listdir(os.path.join(CAR_FOUNDATION_DATA_DIR, 'mujoco_sim_finetune') )
# acc_list = []
# for file in files:
#     d = pickle.load(open(os.path.join(CAR_FOUNDATION_DATA_DIR, 'mujoco_sim_finetune', file), 'rb'))
#     acc_list.append(d.car_params['delay'])  
#     # if acc_list[-1] < 10:
#     #     print("file:", file, "acc:", acc_list[-1])
#     #     os.remove(os.path.join(CAR_FOUNDATION_DATA_DIR, 'numeric_sim', file))
    
# plt.hist(acc_list, bins=100, color='red')


plt.show()