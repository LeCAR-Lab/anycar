import os
import ray
import time
import pickle
import datetime
import numpy as np
from car_dataset import CarDataset
from car_dynamics.envs import make_env, CarEnvParams
from car_dynamics.envs.numeric_sim.car_numeric import Car2D
from car_dynamics.controllers_torch import AltPurePursuitController, RandWalkController
# from car_planner.track_generation_realistic import change_track
from car_planner.track_generation import change_track
from car_planner.global_trajectory import generate_oval_trajectory
from rich.progress import track
import matplotlib.pyplot as plt
from car_foundation import CAR_FOUNDATION_DATA_DIR

RENDER = False
DEBUG = False
CONSTANT_VEL = False

data_folder_prefix = datetime.datetime.now().isoformat(timespec='milliseconds')


ray.init(local_mode=DEBUG)

def show_debug_plots(dataset):
    plt.plot(dataset.data_logs["traj_x"], dataset.data_logs["traj_y"], label='reference')
    plt.plot(dataset.data_logs["xpos_x"], dataset.data_logs["xpos_y"], linestyle = "dashed", label='real')
    plt.title("trajectory vs position")
    dataset.data_logs["xpos_x"] = []
    dataset.data_logs["xpos_y"] = []
    dataset.data_logs["traj_x"] = []
    dataset.data_logs["traj_y"] = []
    plt.legend()
    plt.show()
    
    # plt.plot(np.linalg.norm(np.array([dataset.data_logs["xvel_x"], dataset.data_logs["xvel_y"]]),axis=0))
    # plt.plot(dataset.data_logs["throttle"], label = "target")
    # dataset.data_logs["xvel_z"] = []
    # dataset.data_logs["xvel_x"] = []
    # dataset.data_logs["xvel_y"] = []
    # plt.legend()
    # plt.show()
    
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

@ray.remote   
def rollout(params):
    id, time_steps, debug_plots, datadir = params
    tic = time.time()

    dataset = CarDataset()

    dataset.car_params['sim'] = 'car-numeric'
    
    dataset.car_params['mass'] = np.random.uniform(2.5, 6.)
    # dataset.car_params['mass'] = 4.
    dataset.car_params['friction'] = np.random.uniform(0.4, 1.0)
    # dataset.car_params['friction'] = .8
    dataset.car_params['max_throttle'] = np.random.uniform(5., 20.)
    # dataset.car_params['max_throttle'] = 16.
    dataset.car_params['delay'] = np.random.randint(0, 6)
    # dataset.car_params['delay'] = 0
    dataset.car_params["max_steer"] = np.random.uniform(0.2, 0.38)
    # dataset.car_params["max_steer"] = 0.36
    dataset.car_params["steer_bias"] = np.random.uniform(-0.02, 0.02)
    # dataset.car_params["steer_bias"] = 0.
    
    dataset.car_params["wheelbase"] = np.random.uniform(0.2, 0.5)
    # dataset.car_params['wheelbase'] = 0.31
    dataset.car_params["com"] = np.random.uniform(0.3, 0.7)
    
    env_param = CarEnvParams(
        name='car-numeric-2d',
        mass=dataset.car_params["mass"],
        friction=dataset.car_params["friction"],
        render=RENDER,
        delay=dataset.car_params["delay"],
        max_throttle=dataset.car_params['max_throttle'],
        max_steer = dataset.car_params['max_steer'],
        steer_bias = dataset.car_params['steer_bias'],
        wheelbase = dataset.car_params['wheelbase'],
        com=dataset.car_params['com'],
    )
    
    env = make_env(env_param)
    print("Car Params", dataset.car_params)
    dataset.car_params["wheelbase"] = env.wheelbase
    track_change_time = time_steps
    
    # pure pursuit controller
    ppcontrol = AltPurePursuitController({
        'wheelbase': dataset.car_params["wheelbase"], 
        'totaltime': track_change_time,
        'max_steering': dataset.car_params['max_steer'],
        'lowervel': 0.,
        'uppervel': 4.,
    })

    # random walker
    randcontrol = RandWalkController(track_change_time)
    
    log_timesteps = time_steps

    # trajectory = change_track(scale=1, direction=1)
    # plt.plot(trajectory[:, 0], trajectory[:, 1])
    # plt.show()

    rollout_counter = 0
    
    assert (time_steps % track_change_time == 0)
    assert (time_steps % log_timesteps == 0)

    kp = np.random.uniform(6, 10)
    kd = np.random.uniform(0.5, 1.5)
    # kp = 8
    # kd = 1.
    target_vel_list = []
    real_vel_list = []
    
    # all_controllers = [ppcontrol, randcontrol]
    all_controllers = [ppcontrol, ppcontrol]
    
    
    trajectory = change_track(scale=1, direction=np.random.choice([-1, 1]))
    
    ## overfit oval trajectory
    # raidus_1 = np.random.uniform(1., 1.8)
    # raidus_2 = np.random.uniform(1., 1.8)
    # trajectory = generate_oval_trajectory((0., 0.), raidus_1, raidus_2, direction=np.random.choice([-1, 1]), endpoint=True)
    
    controller = all_controllers[np.random.choice([0, 1])]
    controller.reset_controller()
    env.reset()
    last_err_vel = 0.
    
    target_vel_constant = np.random.uniform(0.5, 3.)
    
    for t in track(range(time_steps), disable=not DEBUG):
        action = controller.get_control(t%track_change_time, env, trajectory)
        if controller.name == "random_walk":
            target_vel = action[0]
        elif controller.name == "pure_pursuit":
            if CONSTANT_VEL:
                target_vel = target_vel_constant
            else:
                target_vel = action[0]
            action[0] = kp * (target_vel - env.car_lin_vel[0]) + kd * ((target_vel - env.car_lin_vel[0]) - last_err_vel)
            # action[1] = 1.
            action[0] /= env.sim.params.Ta
            last_err_vel = target_vel - env.car_lin_vel[0]
        else:
            raise ValueError(f"Unknown Controller: {controller.name}")
        # print(action[0])
        
        target_vel_list.append(target_vel)
        real_vel_list.append(env.car_lin_vel[0])
        action = np.clip(action, env.action_space.low, env.action_space.high)
        log_data(dataset, env, action, controller)
        
        obs, reward, done, info = env.step(np.array(action))
        
        # Chnage Track every Track change timesteps
        if ((t+1) % track_change_time == 0):
            # print(t+1)
            # print(t)
            dataset.data_logs["lap_end"][-1] = 1 # log the end of the lap
            rollout_counter += 1

            trajectory = change_track(scale=1, direction=np.random.choice([-1, 1]))

            controller = all_controllers[np.random.choice([0, 1])]
            controller.reset_controller()
            env.reset()
            

            if debug_plots:
                show_debug_plots(dataset)
                plt.figure()
                plt.plot(target_vel_list, label='target')
                plt.plot(real_vel_list, label='real')
                plt.legend()
                plt.show()
                
            last_err_vel = 0.

        # Log data every log_timesteps 
        if (t+1) % log_timesteps == 0:
            now = datetime.datetime.now().isoformat(timespec='milliseconds')
            file_name = "log_" + str(id) + '_' + str(now) + ".pkl"
            filepath = os.path.join(datadir, file_name)
            
            for key, value in dataset.data_logs.items():
                dataset.data_logs[key] = np.array(value)

            with open(filepath, 'wb') as outp: 
                pickle.dump(dataset, outp, pickle.HIGHEST_PROTOCOL)

            print("Saved Data to:", filepath)

            dataset.reset_logs()
            
    print("Simulation Complete!")
    print("Total Timesteps: ", simend + 1)
    print("Elapsed_Time: ", time.time() - tic)
    print("Total Laps", rollout_counter*25)

if __name__ == "__main__":
    
    simend = 2000
    episodes = 1

    data_dir = os.path.join(CAR_FOUNDATION_DATA_DIR, f'{data_folder_prefix}-numeric_sim')

    os.makedirs(data_dir, exist_ok=True)
    
    if not DEBUG:
        futures = [rollout.remote((i, simend, DEBUG, data_dir)) for i in range(episodes)]
        done = [] 
        # Function to track progress
        def track_progress(futures):
            while len(futures) > 0:
                done, futures = ray.wait(futures, num_returns=1, timeout=1.0)
                for _ in done:
                    yield

        # Use rich.progress.track to display progress
        for _ in track(track_progress(futures), 
                    description="Collecting data...", total=len(futures),disable=False):
            pass

        # Collect the results from workers
        results = ray.get(futures + done)
    else:
        for i in range(1):
            rollout((i, simend, DEBUG, data_dir))
            print(f"Episode {i} Complete")