from car_foundation import CAR_FOUNDATION_DATA_DIR
from car_dynamics import ASSETTO_CORSA_ASSETS_DIR
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
from car_dataset import CarDataset


from car_dynamics.envs.assetto_corsa.assetto_corsa_gym.AssettoCorsaEnv import assettoCorsa #capital C in assetto"C"orsa!!
from car_dynamics.controllers_torch import AltPurePursuitController, RandWalkController

import sys
import pandas as pd
import glob as glob
from omegaconf import OmegaConf

import faulthandler
faulthandler.enable()

def log_data(dataset, env, controller, action):
        dataset.data_logs["xpos_x"].append(env.state["world_position_x"])
        dataset.data_logs["xpos_y"].append(env.state["world_position_y"])
        dataset.data_logs["xpos_z"].append(env.state["world_position_z"])
        #log orientation
        roll, pitch, yaw = env.state["roll"], env.state["pitch"], env.state["yaw"]
        quat = R.from_euler("xyz", [roll, pitch, yaw]).as_quat()
        dataset.data_logs["xori_w"].append(quat[3])
        dataset.data_logs["xori_x"].append(quat[0])
        dataset.data_logs["xori_y"].append(quat[1])
        dataset.data_logs["xori_z"].append(quat[2])
        #log linear velocity
        dataset.data_logs["xvel_x"].append(env.state["local_velocity_x"])
        dataset.data_logs["xvel_y"].append(env.state["local_velocity_y"])
        dataset.data_logs["xvel_z"].append(env.state["local_velocity_z"])
        #log linear acceleration
        dataset.data_logs["xacc_x"].append(env.state["accelX"])
        dataset.data_logs["xacc_y"].append(env.state["accelY"])
        dataset.data_logs["xacc_z"].append(0)
        #log angular velocity
        dataset.data_logs["avel_x"].append(env.state["angular_velocity_x"])
        dataset.data_logs["avel_y"].append(env.state["angular_velocity_y"])  
        dataset.data_logs["avel_z"].append(env.state["angular_velocity_z"])

        dataset.data_logs["traj_x"].append(controller.target_pos[0])
        dataset.data_logs["traj_y"].append(controller.target_pos[1])

        dataset.data_logs["lap_end"].append(0)

        dataset.data_logs["throttle"].append(action[0])
        dataset.data_logs["steer"].append(action[1])

def rollout(id, simend, debug_plots, datadir):
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format of the log messages
        datefmt='%Y-%m-%d %H:%M:%S',  # Format of the timestamp
    )

    tic = time.time()

    dataset = CarDataset()
    config = OmegaConf.load(os.path.join(ASSETTO_CORSA_ASSETS_DIR, "config.yml"))
    env = assettoCorsa.make_ac_env(cfg=config, work_dir="output")

    static_info = env.client.simulation_management.get_static_info()
    ac_mod_config = env.client.simulation_management.get_config()

    logger.info("Static info:")
    for i in static_info:
        logger.info(f"{i}: {static_info[i]}")
    logger.info("AC Mod config:")
    for i in ac_mod_config:
        logger.info(f"{i}: {ac_mod_config[i]}")

    env.reset()

    # set the simulator
    dataset.car_params["sim"] = "assetto_corsa_"+ env.car_name #env.name
    # set the wheelbase
    dataset.car_params["wheelbase"] = env.wheelbase
    # generate a new mass
    dataset.car_params["mass"] = None #env.generate_new_mass()
    # generate a new com
    dataset.car_params["com"] = None #env.generate_new_com()
    # generate a new friction
    dataset.car_params["friction"] = None#env.generate_new_friction()
    # generate new max throttle
    dataset.car_params["max_throttle"] = None#env.generate_new_max_throttle()
    # generate new delay
    dataset.car_params["delay"] = None#env.generate_new_delay()
    # generate new max steering
    dataset.car_params["max_steer"] = None#env.generate_new_max_steering()
    # generate new steering bias
    dataset.car_params["steer_bias"] = None#env.generate_new_steering_bias()

    print("Car Params", dataset.car_params)

    #wenli where did you get the lower vel from?
    ppcontrol = AltPurePursuitController({
        'wheelbase': dataset.car_params["wheelbase"], 
        'totaltime': simend,
        'lowervel': 10, #actual min vel
        'uppervel': 30., #actual max vel   
        'max_steering': 0.61 #about 35 degrees  
    })

    controller = ppcontrol #all_controllers[np.random.choice([0, 1])]
    trajectory = np.array([env.ref_lap.df["pos_x"], env.ref_lap.df["pos_y"]]).T

    # tuned kp and kd
    kp = 1
    kd = 0
    
    last_err_vel = 0.
    is_terminate = False 
    clipped = 0
    actions = []
    vels = []
    for t in range(simend):
        start = time.time()
        action = controller.get_control(t, env, trajectory)
        if controller.name == "pure_pursuit":

            target_vel = action[0]

            action[0] = kp * (target_vel - env.lin_vel[0]) + kd * ((target_vel - env.lin_vel[0]) - last_err_vel)
            
            # print(action[0])
            vels.append(env.lin_vel[0])
            last_err_vel = target_vel - env.lin_vel[0]
        else:
            raise ValueError(f"Unknown Controller: {controller.name}")

        # action = [0., 0.] #throttle. steer
        print("time taken:", time.time() - start)
        action = np.clip(action, -1, 1)
        # actions.append(action[0]) 
        log_data(dataset, env, controller, action)

        obs, reward, done, info = env.step(np.array(action))

        # check if the robot screwed up
        #TODO Fix when adding slope changes to the world.
        if done:
            is_terminate = True
            break
    
    if not is_terminate:
        # dataset.data_logs["lap_end"][-1] = 1 
        now = datetime.datetime.now().isoformat(timespec='milliseconds')
        file_name = "log_" + str(id) + '_' + str(now) + ".pkl"
        filepath = os.path.join(datadir, file_name)
        
        for key, value in dataset.data_logs.items():
            dataset.data_logs[key] = np.array(value)

        # with open(filepath, 'wb') as outp: 
        #     pickle.dump(dataset, outp, pickle.HIGHEST_PROTOCOL)

        print("Saved Data to:", filepath)

    if debug_plots:
        actions = np.array(actions)
        vels = np.array(vels)
        # print(actions.shape)
        # plt.plot(actions, label = "actual command")
        # plt.plot(actions[:, 1], label = "steering")
        plt.plot(ppcontrol.target_velocities, label="target vel")
        plt.plot(vels, label = "actual vel")
        plt.legend()
        plt.show()
        plt.plot(dataset.data_logs["traj_x"], dataset.data_logs["traj_y"], label='Trajectory')
        plt.plot(dataset.data_logs["xpos_x"], dataset.data_logs["xpos_y"], linestyle = "dashed", label='Car Position')
        plt.show()
        
        dataset.reset_logs()
            
    print("Simulation Complete!")
    print("Total Timesteps: ", simend + 1)
    print("Elapsed_Time: ", time.time() - tic)

    env.recover_car()
    env.close()
    return not is_terminate

if __name__ == "__main__":

    debug_plots = True
    simend = 1000
    episodes = 1
    data_dir = os.path.join(CAR_FOUNDATION_DATA_DIR, "assetto_corsa_sim_debugging")
    os.makedirs(data_dir, exist_ok=True)

    num_success = 0
    start = time.time()
    
    for i in range(episodes):
        ret = rollout(i, simend, debug_plots, data_dir)
        print(f"Episode {i} Complete")
        if ret:
            num_success += 1
    
    dur = time.time() - start
    print(f"Success Rate: {num_success}/{episodes}")
    print(f"Time Elapsed:, {dur}")   