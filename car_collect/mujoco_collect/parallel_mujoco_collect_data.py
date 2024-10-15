from car_foundation import CAR_FOUNDATION_DATA_DIR
import numpy as np
import os
import pickle
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import ray
import random
import time
import math
import datetime
from tqdm import tqdm
from car_dynamics.envs.mujoco_sim.cam_utils import *
from car_dataset import CarDataset
from car_planner.track_generation import change_track
# from car_planner.track_generation_realistic import change_track
from car_dynamics import MUJOCO_MODEL_DIR

from car_dynamics.envs.mujoco_sim.car_mujoco import MuJoCoCar
from car_dynamics.controllers_torch import AltPurePursuitController, RandWalkController

import faulthandler
faulthandler.enable()

def log_data(dataset, env, action, controller):
        dataset.data_logs["xpos_x"].append(env.world.pose[0])
        dataset.data_logs["xpos_y"].append(env.world.pose[1])
        dataset.data_logs["xpos_z"].append(env.world.pose[2])
        #log orientation
        dataset.data_logs["xori_w"].append(env.world.orientation[0])
        dataset.data_logs["xori_x"].append(env.world.orientation[1])
        dataset.data_logs["xori_y"].append(env.world.orientation[2])
        dataset.data_logs["xori_z"].append(env.world.orientation[3])
        #log linear velocity
        dataset.data_logs["xvel_x"].append(env.world.lin_vel[0])
        dataset.data_logs["xvel_y"].append(env.world.lin_vel[1])
        dataset.data_logs["xvel_z"].append(env.world.lin_vel[2])
        #log linear acceleration
        dataset.data_logs["xacc_x"].append(env.world.lin_acc[0])
        dataset.data_logs["xacc_y"].append(env.world.lin_acc[1])
        dataset.data_logs["xacc_z"].append(env.world.lin_acc[2])
        #log angular velocity
        dataset.data_logs["avel_x"].append(env.world.ang_vel[0])
        dataset.data_logs["avel_y"].append(env.world.ang_vel[1])    

        dataset.data_logs["avel_z"].append(env.world.ang_vel[2])

        dataset.data_logs["traj_x"].append(controller.target_pos[0])
        dataset.data_logs["traj_y"].append(controller.target_pos[1])

        dataset.data_logs["lap_end"].append(0)

        dataset.data_logs["throttle"].append(action[0])
        dataset.data_logs["steer"].append(action[1])

# @ray.remote
def rollout(id, simend, render, debug_plots, datadir):
    tic = time.time()

    dataset = CarDataset()
    env = MuJoCoCar({'is_render': render}) # create the simulation environment
    env.reset()

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

    # fine tuning data collection
    # dataset.car_params['mass'] = 3.794
    # dataset.car_params['friction'] = 1.
    # dataset.car_params['max_throttle'] = 10.
    # dataset.car_params['delay'] = 4

    print("Car Params", dataset.car_params)

    #choose to change parameters or not
    env.world.change_parameters(dataset.car_params, change = False)

    direction = np.random.choice([-1, 1])
    scale = int(np.random.uniform(1, 5))

    #wenli where did you get the lower vel from?
    ppcontrol = AltPurePursuitController({
        'wheelbase': dataset.car_params["wheelbase"], 
        'totaltime': simend,
        'lowervel': 0.7, #actual min vel
        'uppervel': 2., #actual max vel   
        'max_steering': env.world.max_steer    
    })

    controller = ppcontrol #all_controllers[np.random.choice([0, 1])]
    trajectory = change_track(scale, direction)
    
    # tuned kp and kd
    kp = np.random.uniform(6, 10)
    kd = np.random.uniform(0.5, 1.5)
    
    last_err_vel = 0.
    is_terminate = False 
    clipped = 0
    actions = []
    for t in tqdm(range(simend)):

        action = controller.get_control(t, env.world, trajectory)
        if controller.name == "pure_pursuit":

            target_vel = action[0]

            action[0] = kp * (target_vel - env.world.lin_vel[0]) + kd * ((target_vel - env.world.lin_vel[0]) - last_err_vel)

            action[0] /= env.world.max_throttle # normalize to -1, 1
            
            # print(action[0])
            last_err_vel = target_vel - env.world.lin_vel[0]
        else:
            raise ValueError(f"Unknown Controller: {controller.name}")

        prev_action = action
        #TODO Fix this, the action is getting clipped like 25% of the time
        action = np.clip(action, env.action_space.low, env.action_space.high)
        # actions.append(action[0]) 
        log_data(dataset, env, action, controller)

        obs, reward, done, info = env.step(np.array(action))

        # check if the robot screwed up
        #TODO Fix when adding slope changes to the world.
        if np.abs(env.world.rpy[0]) > 0.05 or np.abs(env.world.rpy[1]) > 0.05:
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
            actions = np.array(actions)
            print(actions.shape)
            plt.plot(actions, label = "actual command")
            # plt.plot(actions[:, 1], label = "steering")
            # plt.plot(ppcontrol.target_velocities, label="targetvel")
            plt.legend()
            plt.show()
            plt.plot(dataset.data_logs["traj_x"], dataset.data_logs["traj_y"], label='Trajectory')
            plt.plot(dataset.data_logs["xpos_x"], dataset.data_logs["xpos_y"], linestyle = "dashed", label='Car Position')
            plt.show()
        
        dataset.reset_logs()
            
    print("Simulation Complete!")
    print("Total Timesteps: ", simend + 1)
    print("Elapsed_Time: ", time.time() - tic)
    return not is_terminate

if __name__ == "__main__":

    render = True
    debug_plots = False
    simend = 20000
    episodes = 1
    data_dir = os.path.join(CAR_FOUNDATION_DATA_DIR, "mujoco_sim_debugging")
    os.makedirs(data_dir, exist_ok=True)

    num_success = 0
    start = time.time()
    if render or debug_plots:
        for i in range(episodes):
            ret = rollout(i, simend, render, debug_plots, data_dir)
            print(f"Episode {i} Complete")
            if ret:
                num_success += 1
        
        dur = time.time() - start
        print(f"Success Rate: {num_success}/{episodes}")
        print(f"Time Elapsed:, {dur}")   
    else:
        assert(render == False and debug_plots == False)
        # Let's start Ray
        ray.init()
        ret = []
        for i in range(episodes):
            ret.append(rollout.remote(i, simend, render, debug_plots, data_dir))
            print(f"Episode {i} Complete")
        output = ray.get(ret)
        # print(output)
        for ifsuccess in output:
            if ifsuccess:
                num_success+=1
        dur = time.time() - start
        print(f"Success Rate: {num_success}/{episodes}")
        print(f"Time Elapsed:, {dur}")    
