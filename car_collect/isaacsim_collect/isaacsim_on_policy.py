from car_foundation import CAR_FOUNDATION_DATA_DIR
import numpy as np
import os
import pickle
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from car_dynamics.envs.isaac_sim.car_isaac import IsaacCar
from car_dataset import CarDataset
from car_dynamics.controllers_torch import AltPurePursuitController, RandWalkController
from car_planner.global_trajectory import GlobalTrajectory, generate_circle_trajectory, generate_oval_trajectory, generate_rectangle_trajectory, generate_raceline_trajectory
from car_planner.track_generation import change_track
import time
import datetime
from tqdm import tqdm
from car_ros2.utils import load_mppi_params, load_dynamic_params
from car_dynamics.controllers_jax import MPPIController, rollout_fn_jax, MPPIRunningParams
from car_dynamics.models_jax import DynamicsJax
from car_foundation import CAR_FOUNDATION_MODEL_DIR
import jax
from isaacsim_collect import ISAACSIM_COLLECT_TMP_DIR
import faulthandler
import jax.numpy as jnp
import socket
import pickle
from termcolor import colored

data_folder_prefix = datetime.datetime.now().isoformat(timespec='milliseconds')
data_folder_path = os.path.join(CAR_FOUNDATION_DATA_DIR, f'{data_folder_prefix}-on-policy-dataset')
cache_folder_path = os.path.join(ISAACSIM_COLLECT_TMP_DIR, f'{data_folder_prefix}-on-policy-cache')    

faulthandler.enable()



def log_data(dataset, env, action,):
        dataset.data_logs["xpos_x"].append(env.full_obs[0])
        dataset.data_logs["xpos_y"].append(env.full_obs[1])
        dataset.data_logs["xpos_z"].append(env.full_obs[2])
        #log orientation
        dataset.data_logs["xori_w"].append(env.full_obs[3])
        dataset.data_logs["xori_x"].append(env.full_obs[4])
        dataset.data_logs["xori_y"].append(env.full_obs[5])
        dataset.data_logs["xori_z"].append(env.full_obs[6])
        #log linear velocity
        dataset.data_logs["xvel_x"].append(env.full_obs[7])
        dataset.data_logs["xvel_y"].append(env.full_obs[8])
        dataset.data_logs["xvel_z"].append(env.full_obs[9])
        #log linear acceleration
        dataset.data_logs["xacc_x"].append(env.full_obs[10])
        dataset.data_logs["xacc_y"].append(env.full_obs[11])
        dataset.data_logs["xacc_z"].append(env.full_obs[12])
        #log angular velocity
        dataset.data_logs["avel_x"].append(env.full_obs[13])
        dataset.data_logs["avel_y"].append(env.full_obs[14])
        dataset.data_logs["avel_z"].append(env.full_obs[15])


        dataset.data_logs["lap_end"].append(0)

        dataset.data_logs["throttle"].append(action[0])
        dataset.data_logs["steer"].append(action[1])

# @ray.remote
def rollout(id, simend, render, debug_plots, datadir, env):
    tic = time.time()

    dataset = CarDataset()
    dataset.car_params['sim'] = "isaac"
    
    tmp_dir = os.path.join(cache_folder_path, f"setting-")
    
    # set the simulator
    dataset.car_params["sim"] = env.name
    # set the wheelbase
    dataset.car_params["wheelbase"] = env.wheelbase
    # generate a new mass
    dataset.car_params["mass"] = env.generate_new_mass()
    # generate a new com
    dataset.car_params["com"] = env.generate_new_com()
    # generate a new friction
    dataset.car_params["friction"] = env.generate_new_friction()
    # generate new max throttle
    dataset.car_params["max_throttle"] = env.generate_new_max_throttle()
    # generate new delay
    dataset.car_params["delay"] = env.generate_new_delay()
    # generate new max steering
    dataset.car_params["max_steer"] = env.generate_new_max_steering()
    # generate new steering bias
    dataset.car_params["steer_bias"] = env.generate_new_steering_bias()

    # fine tuningresume_model_folder_path
    # dataset.car_params['mass'] = None
    # dataset.car_params['friction'source /home/lecar/.local/share/ov/pkg/isaac-sim-2023.1.1/setup_conda_env.sh] = None
    # dataset.car_params['max_throttle'] = None
    # dataset.car_params['delay'] = 0

    print("Car Params", dataset.car_params)

    env.change_parameters(dataset.car_params, change = False)
    
    real_trajectory = []
    
    is_terminate = False
    actions = []
    host = '127.0.0.1'
    port = 65432
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Server listening on {host}:{port}")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            for t in range(simend):
                state = [t, env.obs_state()]
                conn.sendall(pickle.dumps(state))
                action_rec = conn.recv(4096)
                if not action_rec:
                    break
                action = pickle.loads(action_rec)
                print(f"Received Action: {action}")
            
                # action = np.array([0., 0.])

                log_data(dataset, env, action)
                obs, reward, done, info = env.step(np.array(action))
                
                # mppi_running_params = mppi.feed_hist(mppi_running_params, state, action)
                real_trajectory.append(obs[:2])
                
                if np.abs(env.rpy[0]) > 0.05 or np.abs(env.rpy[1]) > 0.05:
                    is_terminate = True
                    break
                
            if not is_terminate:
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
                    plt.figure()
                    plt.plot(np.array(real_trajectory)[:, 0], np.array(real_trajectory)[:, 1], label='real', marker='o', markersize=3)
                    # plt.plot(np.array(reference_track)[:, 0], np.array(reference_track)[:, 1], label='reference')
                    plt.legend()
                    plt.savefig(os.path.join(tmp_dir, f"trajectory-{id}.png"))
                
                dataset.reset_logs()
            else:
                print(colored("Simulation Failed"))

            print("Simulation Complete!")
            print("Total Timesteps: ", simend + 1)
            print("Elapsed_Time: ", time.time() - tic)
            return not is_terminate
    

if __name__ == "__main__":

    render = True
    debug_plots = False
    simend = 20000 #1250 for 25 seconds of driving
    episodes = 1
    data_dir = os.path.join(CAR_FOUNDATION_DATA_DIR, "isaac_sim_trash")
    os.makedirs(data_dir, exist_ok=True)
    num_success = 0
    env = IsaacCar({'is_render': render})
    start = time.time()
    for i in range(episodes):
       
        ret = rollout(i, simend, render, debug_plots, data_dir, env)
        env.reset()
        print(f"Episode {i} Complete")
        if ret:
            num_success += 1
    dur = time.time() - start
    print(f"Success Rate: {num_success}/{episodes}")
    print(f"Time Elapsed:, {dur}")    
    env.shutdown()
    print("shutdown complete")
    
        
