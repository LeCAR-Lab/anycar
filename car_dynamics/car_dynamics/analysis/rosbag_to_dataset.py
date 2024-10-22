import os
import numpy as np
import pickle
from car_ros2.bag_utils import BagReader, synchronize_time
import matplotlib.pyplot as plt
import datetime
from car_dataset import CarDataset
from car_foundation import CAR_FOUNDATION_DATA_DIR

data_folder_prefix = datetime.datetime.now().isoformat(timespec='milliseconds')
data_folder_path = os.path.join(CAR_FOUNDATION_DATA_DIR, f'{data_folder_prefix}-rosbag-dataset')

bag_path = "PATH-TO-ROSBAG.mcap"

bag = BagReader(bag_path, full_state=True)

MAX_EPISODE_LENGTH = 2000



action_list = bag.bag_dict['/ackermann_command']
state_list = bag.bag_dict['/odometry_copy']

print("before sync", len(state_list))
action_list = action_list[10:]

actions = [a for t, a in action_list]
states = [s for t, s in state_list]

t_actions = np.array([t for t, a in action_list])
t_states = np.array([t for t, s in state_list])



t_states_new, states_new = synchronize_time(t_states, t_actions, states)


assert states_new.shape[1] == 9    
dataset = CarDataset()

dataset.car_params['sim'] = 'car-real'

print("States shape", states_new.shape)
i = 0
for state, action in zip(states_new, actions):
    dataset.data_logs["xpos_x"].append(state[0])
    dataset.data_logs["xpos_y"].append(state[1])
    dataset.data_logs["xpos_z"].append(0.)
    #log orientation
    dataset.data_logs["xori_w"].append(state[5])
    dataset.data_logs["xori_x"].append(state[2])
    dataset.data_logs["xori_y"].append(state[3])
    dataset.data_logs["xori_z"].append(state[4])
    #log linear velocity
    dataset.data_logs["xvel_x"].append(state[6])
    dataset.data_logs["xvel_y"].append(state[7])
    dataset.data_logs["xvel_z"].append(0.)
    #log linear acceleration
    dataset.data_logs["xacc_x"].append(0.)
    dataset.data_logs["xacc_y"].append(0.)
    dataset.data_logs["xacc_z"].append(0.)
    #log angular velocity
    dataset.data_logs["avel_x"].append(0.)
    dataset.data_logs["avel_y"].append(0.)
    dataset.data_logs["avel_z"].append(state[8])

    dataset.data_logs["traj_x"].append(0.)
    dataset.data_logs["traj_y"].append(0.)

    dataset.data_logs["lap_end"].append(0)

    dataset.data_logs["throttle"].append(action[0])
    dataset.data_logs["steer"].append(action[1])
    
    i += 1
    
    if i == MAX_EPISODE_LENGTH:
        data_dir = os.path.join(data_folder_path)
            
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        dataset.data_logs["lap_end"][-1] = 1    
        now = datetime.datetime.now().isoformat(timespec='milliseconds')
        file_name = "log_" + str(now) + ".pkl"
        filepath = os.path.join(data_dir, file_name)
        for key, value in dataset.data_logs.items():
            dataset.data_logs[key] = np.array(value)

        with open(filepath, 'wb') as outp: 
            pickle.dump(dataset, outp, pickle.HIGHEST_PROTOCOL)
        print("Saved Data to:", filepath)
        
        i = 0
        dataset.reset_logs()
        

for i in range(states_new.shape[0]-1):
    if states_new[i, 0] == states_new[i+1, 0]:
        print(i)
plt.figure()
# plt.plot(states_new[:, 0], states_new[:, 1])
plt.plot(states_new[:, 0], 'r', label='x', marker='o')
plt.plot(states_new[:, 1], 'g', label='y', marker='o')
plt.title("trajectory")
plt.legend()

plt.figure()
plt.plot(states_new[:, 6], 'r', label='x', marker='o')
plt.title("linear velocity vx")

# plot vy
plt.figure()
plt.plot(states_new[:, 7], 'r', label='y', marker='o')
plt.title("linear velocity vy")

# plot ang_vel
plt.figure()
plt.plot(states_new[:, 8], 'r', label='z', marker='o')
plt.title("angular velocity")


plt.show()