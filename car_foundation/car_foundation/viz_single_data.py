import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from car_foundation import CAR_FOUNDATION_DATA_DIR
from car_foundation.dataset import MujocoDataset
from car_dataset import CarDataset
from car_foundation.utils import quaternion_to_euler, generate_subsequences, generate_subsequences_hf, align_yaw
import glob


DATASET_NAME = "XXX"

path = os.path.join(CAR_FOUNDATION_DATA_DIR, DATASET_NAME)
filenumber = 0
print(glob.glob(os.path.join(path, '*.pkl'))[filenumber])
dataset = pickle.load(open(glob.glob(os.path.join(path, '*.pkl'))[filenumber], 'rb'))
MujocoDataset(os.path.join(CAR_FOUNDATION_DATA_DIR, 'mujoco_sim_scale_1'), 50, 50, teacher_forcing=False)
throttle = dataset.data_logs["throttle"]
steer = dataset.data_logs['steer']
q = np.array([dataset.data_logs["xori_w"],
                    dataset.data_logs["xori_x"],
                    dataset.data_logs["xori_y"],
                    dataset.data_logs["xori_z"]]).T
_, _, yaw = quaternion_to_euler(q)
vx = dataset.data_logs["xvel_x"]
vy = dataset.data_logs["xvel_y"]
vyaw = dataset.data_logs["avel_z"]

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


plt.figure()
plt.plot(vx)
plt.show()


# visualize the 2D heatmap of the throttle and steer
plt.figure()
plt.hist2d(throttle.flatten(), steer.flatten(), bins=100, cmap='viridis')
plt.xlim([-1., 1.])
plt.ylim([-1., 1.])
plt.colorbar()
plt.xlabel('Throttle')
plt.ylabel('Steer')
plt.title('Throttle and Steer Distribution')
# plt.savefig('throttle_steer_distribution.png')
plt.show()

# plt.figure()
# plt.hist(throttle.flatten(), bins=1000, color='red')
# plt.title('Throttle Distribution')



plt.figure()
plt.hist2d(vx.flatten(), vy.flatten(), bins=100, cmap='viridis')
plt.colorbar()
plt.xlabel('Vx')
plt.ylabel('Vy')
plt.title('Vx and Vy Distribution')
# plt.savefig('vx_vy_distribution.png')
plt.show()
# plt.savefig('throttle_steer_distribution.png')

# plt.show()

plt.figure()
plt.hist(vx.flatten(), bins=100, color='red')
plt.xlabel('Linear Velocity x')
plt.ylabel('Frequency')
# plt.plot(lin_vel_x.flatten())
plt.title('Linear Velocity x Distribution')
plt.show()

plt.figure()
plt.hist(vy.flatten(), bins=100, color='red')
plt.xlabel('Linear Velocity y')
plt.ylabel('Frequency')
plt.title('Linear Velocity y Distribution')
plt.show()

quit()