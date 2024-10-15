import socket
import pickle
import os
from car_foundation import CAR_FOUNDATION_MODEL_DIR
from car_planner.track_generation import change_track
from car_dynamics.models_jax import DynamicBicycleModel
from car_dynamics.controllers_jax import MPPIController, rollout_fn_select, MPPIRunningParams, void_fn
from car_planner.global_trajectory import GlobalTrajectory, generate_circle_trajectory, generate_oval_trajectory, generate_rectangle_trajectory, generate_raceline_trajectory
from car_dynamics.controllers_jax import MPPIController, rollout_fn_jax, MPPIRunningParams
from car_ros2.utils import load_mppi_params, load_dynamic_params
from car_dynamics.controllers_torch import PurePersuitParams, PurePersuitController, AltPurePursuitController
from scipy.spatial.transform import Rotation as R
from car_dynamics.models_jax import DynamicsJax
from termcolor import colored
import numpy as np
import jax
import jax.numpy as jnp
from pynput import keyboard
import datetime
import time
import matplotlib.pyplot as plt
import math

def fn():
    ...
    
    
load_checkpoint = True
resume_model_checkpint = 120
# resume_model_name = "2024-07-15Thistory_length = 251
# resume_model_name = "2024-07-20T12:56:00.490-model_checkpoint"
# resume_model_name = "2024-07-17T22:47:11.861-model_checkpoint"
resume_model_name = "2024-07-21T15_51_52.343-model_checkpoint"
resume_model_folder_path = os.path.join(CAR_FOUNDATION_MODEL_DIR, resume_model_name, f"{resume_model_checkpint}", "default")
# controller_type = 'mppi'
# controller_type = 'pure_persuit'
history_length = 251
prediction_length = 50
mppi_dual = True
use_recover_controller = False
print(resume_model_folder_path)

def on_press_key(key):
        global use_recover_controller
        # print(key, type(key), dir(key), key.char)
        if hasattr(key, 'char') and key.char == 'r':
            use_recover_controller = not use_recover_controller
            print(colored(f"[INFO] Recover mode: {use_recover_controller}", "blue"))
        # if hasattr(key, 'char') and key.char == 'q':
        #     self.emergency_stop = not self.emergency_stop
        #     if self.emergency_stop:
        #         print(colored(f"[INFO] Emergency stop", "red"))

def start_controller(host='localhost', port=65432):
    global use_recover_controller

    jax_key = jax.random.PRNGKey(123)
    jax_key, key2 = jax.random.split(jax_key, 2)
    jax_key = key2

    #TODO Load in the right track using Pickle File!!!
    filepath = "ref_trajectories/rbring_gp.pkl"
    with open(filepath, "rb") as input_file:
        reference_track = pickle.load(input_file)
    reference_track = reference_track[::4, :]
    # print(reference_track.shape)
    # print(reference_track)
    # plt.plot(reference_track[:,0], reference_track[:,1])
    # plt.savefig("track.png")
    # plt.clf()

    global_planner = GlobalTrajectory(reference_track)
    
    ## Load mppi model
    mppi_params = load_mppi_params()
    model_params = load_dynamic_params()
    model_params.LF = 1.4
    model_params.LR = 1.35
    model_params.MASS = 580
    model_params.Ta = 14.715
    model_params.Sa = 0.61

    L = model_params.LF + model_params.LR
    
    dynamics = DynamicsJax({'model_path': resume_model_folder_path})
    print(colored("Loaded transformer model", "green"))
    print(colored(type(dynamics), "blue"))
    
    rollout_fn = rollout_fn_jax(dynamics)
    
    jax_key, key2 = jax.random.split(jax_key)
    
    mppi = MPPIController(
        mppi_params, rollout_fn, fn, key2
    )
    
    mppi_running_params = mppi.get_init_params()
    
    mppi_running_params = MPPIRunningParams(
        a_mean = mppi_running_params.a_mean,
        a_cov = mppi_running_params.a_cov,
        prev_a = mppi_running_params.prev_a,
        state_hist = mppi_running_params.state_hist,
        key = key2,
    )

    # MPPI Warmup
    ## Define the warmup MPPI Based on DBM model
    mppi_params_warmup = load_mppi_params()
    mppi_params_warmup.dynamics = 'dbm'
    dynamics_warmup = DynamicBicycleModel(model_params)
    L = model_params.LF + model_params.LR
    dynamics_warmup.reset()
    rollout_fn_warmup = rollout_fn_select('dbm', dynamics_warmup, model_params.DT, L, model_params.LR)
    jax_key, key2 = jax.random.split(jax_key, 2)
    mppi_warmup = MPPIController(mppi_params_warmup, rollout_fn_warmup, void_fn, key2)
    mppi_running_params_warmup = mppi_warmup.get_init_params()
    jax_key, key2 = jax.random.split(jax_key, 2)
    mppi_running_params_warmup = MPPIRunningParams(
        a_mean = mppi_running_params_warmup.a_mean,
        a_cov = mppi_running_params_warmup.a_cov,
        prev_a = mppi_running_params_warmup.prev_a,
        state_hist = mppi_running_params_warmup.state_hist,
        key = key2,
    )

    pure_persuit_params = PurePersuitParams()
    pure_persuit_params.mode = 'throttle'
    pure_persuit_params.target_vel = 10
    pure_persuit_params.wheelbase = 2.79
    # pure_persuit_params.wheelbase = 0.31
    pure_persuit_params.max_steering_angle = 0.61
    pure_persuit_params.kp = 1.
    recover_controller = PurePersuitController(pure_persuit_params)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        print(f"Connected to server at {host}:{port}")
        while True:
            data = s.recv(4096)
            [t, state]= pickle.loads(data)
            # print(f"Received data: {[t, state]}")
            
            pose_car = (state[0], state[1], 0)
            lin_vel_car = (state[3], state[4], 0)
            r = R.from_euler('xyz', (0, 0, state[2]))
            quat_car = r.as_quat()
            quat_car = np.array([quat_car[3], quat_car[0], quat_car[1], quat_car[2]])

            if t == 0: # warm up the history
                for _ in range(history_length):
                    mppi_running_params = mppi.feed_hist(mppi_running_params, state, np.array([0., 0.]))

            # action = np.clip(action, env.action_space.low, env.action_space.high)

            target_pos_arr, frenet_pose = global_planner.generate(state[:5], 0.02, (mppi_params.h_knot - 1) * mppi_params.num_intermediate + 2 + mppi_params.delay, True)
            target_pos_list = np.array(target_pos_arr)
            target_pos_tensor = jnp.array(target_pos_arr)
            dynamic_params_tuple = (model_params.LF, model_params.LR, model_params.MASS, model_params.DT, model_params.K_RFY, model_params.K_FFY, model_params.Iz, model_params.Ta, model_params.Tb, model_params.Sa, model_params.Sb, model_params.mu, model_params.Cf, model_params.Cr, model_params.Bf, model_params.Br, model_params.hcom, model_params.fr)
            
            
            
            # action, mppi_running_params, mppi_info = mppi(state,target_pos_tensor,mppi_running_params, dynamic_params_tuple, vis_optim_traj=True,)
            action, mppi_running_params_warmup, mppi_info = mppi(state,target_pos_tensor,mppi_running_params_warmup, dynamic_params_tuple, vis_optim_traj=True,)

            sampled_traj = np.array(mppi_info['trajectory'][:, :2])  
            
            if t == 100:
                plt.plot(target_pos_arr[:, 0], target_pos_arr[:, 1], label = "target")
                plt.plot(sampled_traj[:, 0], sampled_traj[:, 1], label = "sampled")
                plt.plot(state[0], state[1], "x", label = "position")
                plt.legend()
                plt.savefig("debug.png")
                quit()

            if use_recover_controller:
                action = recover_controller.step(pose_car, lin_vel_car, quat_car, target_pos_list)
                if t == 0:
                    action = np.array([0., 0.])
            mppi_running_params = mppi.feed_hist(mppi_running_params, state.copy(), action)
            action = np.array(action, dtype=np.float32)
            s.sendall(pickle.dumps(action))

if __name__ == "__main__":
    listener = keyboard.Listener(on_press=on_press_key)
    listener.start()
    start_controller()
