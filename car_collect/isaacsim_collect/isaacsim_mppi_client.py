import socket
import pickle
import os
from car_foundation import CAR_FOUNDATION_MODEL_DIR
from car_foundation import CAR_FOUNDATION_DATA_DIR
from isaacsim_collect import ISAACSIM_COLLECT_TMP_DIR
from car_planner.track_generation import change_track
from car_dynamics.controllers_torch import AltPurePursuitController, RandWalkController
from car_planner.global_trajectory import GlobalTrajectory, generate_circle_trajectory, generate_oval_trajectory, generate_rectangle_trajectory, generate_raceline_trajectory
from car_dynamics.controllers_jax import MPPIController, rollout_fn_jax, MPPIRunningParams
from car_ros2.utils import load_mppi_params, load_dynamic_params
from car_dynamics.models_jax import DynamicsJax
from termcolor import colored
import numpy as np
import jax
import jax.numpy as jnp
import datetime
import time

def fn():
    ...
    
    
load_checkpoint = True
resume_model_checkpint = 400
resume_model_name = "XXX"
resume_model_folder_path = os.path.join(CAR_FOUNDATION_MODEL_DIR, resume_model_name, f"{resume_model_checkpint}", "default")
controller_type = 'mppi'
history_length = 251
prediction_length = 50

def start_controller(host='127.0.0.1', port=65432):
    jax_key = jax.random.PRNGKey(123)
    jax_key, key2 = jax.random.split(jax_key, 2)
    jax_key = key2
    direction = np.random.choice([-1, 1])
    scale = int(np.random.uniform(1, 2))

    reference_track = change_track(scale, direction)
    global_planner = GlobalTrajectory(reference_track)
    
    ## Load mppi model
    mppi_params = load_mppi_params()
    model_params = load_dynamic_params()
    
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
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        print(f"Connected to server at {host}:{port}")
        while True:
            data = s.recv(4096)
            [t, state]= pickle.loads(data)
            print(f"Received data: {[t, state]}")
            if t == 0:
                for _ in range(history_length):
                    mppi_running_params = mppi.feed_hist(mppi_running_params, state, np.array([0., 0.]))
            # action = np.clip(action, env.action_space.low, env.action_space.high)

            target_pos_arr, frenet_pose = global_planner.generate(state[:5], 0.02, (mppi_params.h_knot - 1) * mppi_params.num_intermediate + 2 + mppi_params.delay, True)
            target_pos_list = np.array(target_pos_arr)
            target_pos_tensor = jnp.array(target_pos_arr)
            dynamic_params_tuple = (model_params.LF, model_params.LR, model_params.MASS, model_params.DT, model_params.K_RFY, model_params.K_FFY, model_params.Iz, model_params.Ta, model_params.Tb, model_params.Sa, model_params.Sb, model_params.mu, model_params.Cf, model_params.Cr, model_params.Bf, model_params.Br, model_params.hcom, model_params.fr)
            print(state)
            action, mppi_running_params, mppi_info = mppi(state,target_pos_tensor,mppi_running_params, dynamic_params_tuple, vis_optim_traj=True,)
            # mppi_time_ = time.time() - st
            # print("time to compute action", time.time() - st)
            st_ = time.time()
            action = np.array(action, dtype=np.float32)
            # action *= 0.
            s.sendall(pickle.dumps(action))

if __name__ == "__main__":
    start_controller()
