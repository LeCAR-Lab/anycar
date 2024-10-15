import numpy as np
from car_planner.track_generation import change_track
import matplotlib.pyplot as plt
import os
from car_dynamics.envs import make_env, CarEnvParams
from car_dataset import CarDataset
from car_planner import CAR_PLANNER_ASSETS_DIR
from car_planner.global_trajectory import GlobalTrajectory, generate_circle_trajectory, generate_oval_trajectory, generate_rectangle_trajectory, generate_raceline_trajectory
from numeric_collect.utils import log_data_minimal, mkdir_if_not_exist
import datetime
from car_foundation import CAR_FOUNDATION_DATA_DIR
import pickle
from car_ros2.utils import load_mppi_params, load_dynamic_params
from car_dynamics.controllers_jax import MPPIController, rollout_fn_jax, MPPIRunningParams
from car_dynamics.models_jax import DynamicsJax
from termcolor import colored
import jax
import time
import jax.numpy as jnp
from numeric_collect import NUMERIC_COLLECT_TMP_DIR
from car_foundation import CAR_FOUNDATION_MODEL_DIR
from car_foundation.jax_models import JaxTransformerDecoder, JaxMLP
import random
import tqdm
from car_foundation.dataset import DynamicsDataset, IssacSimDataset, MujocoDataset
from torch.utils.data import DataLoader
import glob
import optax
import orbax
from flax.training import orbax_utils
from flax.training import train_state
import torch
from car_foundation.utils import generate_subsequences, generate_subsequences_hf, align_yaw, align_yaw_jax
from functools import partial
from rich.progress import track
import ray
from car_planner.track_generation_realistic import change_track


# DEBUG = False

# ray.init(local_mode=DEBUG)
# trajectory = change_track(scale=1, direction=np.random.choice([-1, 1]))
# d = np.loadtxt(os.path.join(CAR_PLANNER_ASSETS_DIR, "15_lecar_optm.txt"))
# import ipdb; ipdb.set_trace()
# plt.plot(trajectory[:, 0], trajectory[:, 1])

# plt.show()


data_folder_prefix = datetime.datetime.now().isoformat(timespec='milliseconds')
data_folder_path = os.path.join(CAR_FOUNDATION_DATA_DIR, f'{data_folder_prefix}-on-policy-dataset')

cache_folder_path = os.path.join(NUMERIC_COLLECT_TMP_DIR, f'{data_folder_prefix}-on-policy-cache')
def fn():
    ...
    

# Load pretrained model
pre_trained_model = ...

num_pickle_files = 10
num_steps = 2000
num_tracks = 4
num_epochs = 2
num_settings = 20
# data collection loop


##  ----------- Model training params ----------------
update_model = False
history_length = 251
prediction_length = 50
delays = None
teacher_forcing = False

FINE_TUNE = False
ATTACK = False
lr_begin = 5e-4
warmup_period = 2
num_train_epochs = 50
load_checkpoint = True
resume_model_checkpint = 400
# resume_model_name = "2024-07-15T17:56:55.014-model_checkpoint"
resume_model_name = "2024-07-17T23:58:15.976-model_checkpoint"
# resume_model_name = "2024-07-17T22:47:11.861-model_checkpoint"
resume_model_folder_path = os.path.join(CAR_FOUNDATION_MODEL_DIR, resume_model_name, f"{resume_model_checkpint}", "default")
val_every = 20
batch_size = 1024
lambda_l2 = 1e-4
comment = 'jax'
num_workers = 6
ATTACK = True
state_dim = 6
action_dim = 2
latent_dim = 64
num_heads = 4
num_layers = 2
dropout = 0.1
PARAMS_KEY = "params"
DROPOUT_KEY = "dropout"
INPUT_KEY = "input_rng"

architecture = 'decoder'
mean = jnp.array([3.3083595e-02, 1.4826456e-04, 1.8982769e-03, 1.6544139e+00, 5.5305376e-03, 9.5738873e-02])
std = jnp.array([0.01598073, 0.00196785, 0.01215522, 0.7989133,  0.09668902, 0.608985])

## ----------------- Helper Functions -----------------------------------

# @ray.remote
def rollout(pickle_i, tmp_dir, mppi, key_i):
    
    track_direction = np.random.choice([-1, 1])
    reference_track = change_track(scale=1, direction=track_direction)
    global_planner = GlobalTrajectory(reference_track)

    env.reset() 
    
    mkdir_if_not_exist(tmp_dir)
    

    mppi_running_params = mppi.get_init_params()
    
    key_i, key2 = jax.random.split(key_i)
    
    mppi_running_params = MPPIRunningParams(
        a_mean = mppi_running_params.a_mean,
        a_cov = mppi_running_params.a_cov,
        prev_a = mppi_running_params.prev_a,
        state_hist = mppi_running_params.state_hist,
        key = key2,
    )
    
    real_trajectory = []
    for t in track(range(num_steps), disable=False):
        state = env.obs_state()
        
        if t == 0:
            for _ in range(history_length):
                mppi_running_params = mppi.feed_hist(mppi_running_params, state, np.array([0., 0.]))
        
        
        target_pos_arr, frenet_pose = global_planner.generate(state[:5], env.sim.params.DT, (mppi_params.h_knot - 1) * mppi_params.num_intermediate + 2 + mppi_params.delay, True)
        target_pos_list = np.array(target_pos_arr)
        target_pos_tensor = jnp.array(target_pos_arr)
        dynamic_params_tuple = (model_params.LF, model_params.LR, model_params.MASS, model_params.DT, model_params.K_RFY, model_params.K_FFY, model_params.Iz, model_params.Ta, model_params.Tb, model_params.Sa, model_params.Sb, model_params.mu, model_params.Cf, model_params.Cr, model_params.Bf, model_params.Br, model_params.hcom, model_params.fr)
        action, mppi_running_params, mppi_info = mppi(state,target_pos_tensor,mppi_running_params, dynamic_params_tuple, vis_optim_traj=True,)
        # mppi_time_ = time.time() - st
        # print("time to compute action", time.time() - st)
        st_ = time.time()
        action = np.array(action, dtype=np.float32)
        # print("DEBUG HIST", mppi_info['history'][-1])

        log_data_minimal(dataset, env, action)
        
        obs, reward, done, info = env.step(np.array(action))
        
        mppi_running_params = mppi.feed_hist(mppi_running_params, state, action)
        real_trajectory.append(obs[:2])
        
    plt.figure()
    plt.plot(np.array(real_trajectory)[:, 0], np.array(real_trajectory)[:, 1], label='real', marker='o', markersize=3)
    plt.plot(np.array(reference_track)[:, 0], np.array(reference_track)[:, 1], label='reference')
    plt.legend()
    plt.savefig(os.path.join(tmp_dir, f"trajectory-{pickle_i}.png"))
    
    dataset.data_logs["lap_end"][-1] = 1 # log the end of the la
    now = datetime.datetime.now().isoformat(timespec='milliseconds')
    file_name = f"dataset-{str(now)}.pkl"
    filepath = os.path.join(data_dir, file_name)

    for key, value in dataset.data_logs.items():
        dataset.data_logs[key] = np.array(value)

    with open(filepath, 'wb') as outp: 
        pickle.dump(dataset, outp, pickle.HIGHEST_PROTOCOL)

    print("Saved Data to:", filepath)

    dataset.reset_logs()
    return True
    

## ----------------- On policy Data Collection Loop ---------------------
jax_key = jax.random.PRNGKey(0)

for setting_i in range(num_settings):
    # Generate track
    dataset = CarDataset()
    
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
    dataset.car_params["steer_bias"] = np.random.uniform(-0.05, 0.05)
    # dataset.car_params["steer_bias"] = 0.
    dataset.car_params["wheelbase"] = np.random.uniform(0.2, 0.5)
    # dataset.car_params['wheelbase'] = 0.31
    dataset.car_params["com"] = np.random.uniform(0.3, 0.7)
    # dataset.car_params["com"] = 0.48
    
    env_param = CarEnvParams(
        name='car-numeric-2d',
        mass=dataset.car_params["mass"],
        friction=dataset.car_params["friction"],
        render=False,
        delay=dataset.car_params["delay"],
        max_throttle=dataset.car_params['max_throttle'],
        max_steer = dataset.car_params['max_steer'],
        steer_bias = dataset.car_params['steer_bias'],
        wheelbase = dataset.car_params['wheelbase'],
        com=dataset.car_params['com'],
    )
    
    env = make_env(env_param)
    
    for track_i in range(num_tracks):
        
        for epoch_i in range(num_epochs):
            
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
            
            # collect on policy data
            data_dir = os.path.join(data_folder_path, f"setting-{setting_i}-track-{track_i}-epoch-{epoch_i}")
            
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            
            
            tmp_dir = os.path.join(cache_folder_path, f"setting-{setting_i}-track-{track_i}-epoch-{epoch_i}")
            
            ## Need to parallelize this
            ret = []
            for pickle_i in range(num_pickle_files):
                jax_key, key2 = jax.random.split(jax_key)
                ret.append(rollout(pickle_i, tmp_dir, mppi, key2))
            # output = ray.get(ret)
                
            
            ## -------------------- train model ---------------------------
            if not update_model:
                continue
            
            dataset_path = data_dir

            # resume_model_folder_path = os.path.join(CAR_FOUNDATION_MODEL_DIR, resume_model_name, f"{resume_model_checkpint}")
            resume_model_folder_path_parent = os.path.join(resume_model_folder_path, "..")
            # resume_model_folder_path = os.path.join(CAR_FOUNDATION_MODEL_DIR, resume_model_name, f"{resume_model_checkpint}", "default")
            

            save_model_folder_prefix = datetime.datetime.now().isoformat(timespec='milliseconds')
            save_model_folder_path = os.path.join(CAR_FOUNDATION_MODEL_DIR, f'{save_model_folder_prefix}-model_checkpoint')

            # architecture = 'mlp'

            # model = TorchTransformer(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, dropout)

            if architecture == 'decoder':
                model = JaxTransformerDecoder(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, dropout, history_length - 1, prediction_length, jnp.bfloat16, name=architecture)
            elif architecture == 'mlp':
                model = JaxMLP([256, 256, 256, 256, 256], state_dim, 0.1, name=architecture)


            # Load the dataset
            binary_mask = False # type(model) == TorchGPT2
            print(colored(f"Loading dataset from: {dataset_path}", "green"))
            dataset_files = glob.glob(os.path.join(dataset_path, '*.pkl'))
            random.shuffle(dataset_files)
            total_len = len(dataset_files)
            split_70 = int(total_len * 0.7)
            split_20 = int(total_len * 0.9)
            data_70 = dataset_files[:split_70]
            data_20 = dataset_files[split_70:split_20]
            data_10 = dataset_files[split_20:]

            train_dataset = MujocoDataset(data_70, history_length, prediction_length, delays=delays, teacher_forcing=teacher_forcing, binary_mask=binary_mask,attack=ATTACK)

            val_dataset = MujocoDataset(data_20, history_length, prediction_length, delays=delays, mean=train_dataset.mean, teacher_forcing=teacher_forcing, std=train_dataset.std, binary_mask=binary_mask, attack=ATTACK)
            test_dataset = MujocoDataset(data_10, history_length, prediction_length, delays=delays, mean=train_dataset.mean, teacher_forcing=teacher_forcing, std=train_dataset.std, binary_mask=binary_mask, attack=ATTACK)

            print("mean: ", mean)
            print("std: ", std)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            num_steps_per_epoch = len(train_loader)

            jax_key, rng = jax.random.split(jax_key)
            rng, params_rng = jax.random.split(rng)
            rng, dropout_rng = jax.random.split(rng)
            init_rngs = {PARAMS_KEY: params_rng, DROPOUT_KEY: dropout_rng}
            global_rngs = init_rngs

            jax_history_input = jnp.ones((batch_size, history_length-1, state_dim + action_dim), dtype=jnp.float32)
            jax_history_mask = jnp.ones((batch_size, (history_length-1) * 2 - 1), dtype=jnp.float32)
            jax_prediction_input = jnp.ones((batch_size, prediction_length, action_dim), dtype=jnp.float32)
            jax_prediction_mask = jnp.ones((batch_size, prediction_length), dtype=jnp.float32)

            def create_learning_rate_fn():
                warmup_fn = optax.linear_schedule(init_value=0.0, end_value=lr_begin, transition_steps=warmup_period)
                decay_fn = optax.exponential_decay(lr_begin, decay_rate=0.99, transition_steps=num_steps_per_epoch, staircase=True)
                schedule_fn = optax.join_schedules(
                    schedules=[warmup_fn, decay_fn],
                    boundaries=[warmup_period]
                )
                return schedule_fn

            learning_rate_fn = create_learning_rate_fn()

            global_var = model.init(init_rngs, jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask)
            tx = optax.adamw(learning_rate_fn, weight_decay=lambda_l2)
            global_state = train_state.TrainState.create(
                        apply_fn=model.apply, params=global_var[PARAMS_KEY], tx=tx
                    )

            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            save_args = orbax_utils.save_args_from_target(global_state)
            options = orbax.checkpoint.CheckpointManagerOptions(create=True, save_interval_steps=val_every)
            print(colored(f"Saving model to: {save_model_folder_path}", "green"))
            checkpoint_manager = orbax.checkpoint.CheckpointManager(save_model_folder_path, orbax_checkpointer, options)

            if load_checkpoint:
                restored = checkpoint_manager.restore(resume_model_folder_path_parent)
                global_var['params'] = restored['params']
                global_state = train_state.TrainState.create(
                        apply_fn=model.apply, params=global_var[PARAMS_KEY], tx=tx
                    )


            def apply_batch(var_collect, last_state, history, action, y, action_padding_mask, rngs):
                history = history.at[:, :, :6].set((history[:, :, :6] - mean) / std)
                y = y.at[:, :, :6].set((y[:, :, :6] - mean) / std)

                x = history[:, 1:, :]
                # tgt_mask = nn.Transformer.generate_square_subsequent_mask(action.size(1), device=action.device)
                y_pred = model.apply(var_collect, x, action, action_padding_mask=action_padding_mask, rngs=rngs, deterministic=True) * std + mean

                last_pose = last_state[:, :3]
                for i in range(y_pred.shape[1]):
                    # rotate dx, dy back to world frame
                    y_pred_x = y_pred[:, i, 0] * jnp.cos(last_pose[:, 2]) - y_pred[:, i, 1] * jnp.sin(last_pose[:, 2])
                    y_pred_y = y_pred[:, i, 0] * jnp.sin(last_pose[:, 2]) + y_pred[:, i, 1] * jnp.cos(last_pose[:, 2])
                    y_pred = y_pred.at[:, i, 0].set(y_pred_x)
                    y_pred = y_pred.at[:, i, 1].set(y_pred_y)
                    # accumulate the poses
                    y_pred = y_pred.at[:, i, :3].add(last_pose)
                    y_pred = y_pred.at[:, i, 2].set(align_yaw_jax(y_pred[:, i, 2], 0.0))
                    last_pose = y_pred[:, i, :3]
                return y_pred

            @partial(jax.jit, static_argnums=(7,))
            def loss_fn(state, var_collect, history, action, y, action_padding_mask, rngs, deterministic=False):
                history = history.at[:, :, :6].set((history[:, :, :6] - mean) / std)
                y = y.at[:, :, :6].set((y[:, :, :6] - mean) / std)
                history = jax.lax.stop_gradient(history[:, 1:, :])
                y = jax.lax.stop_gradient(y)
                action = jax.lax.stop_gradient(action)

                y_pred = state.apply_fn(var_collect, history, action, history_padding_mask=None, action_padding_mask=action_padding_mask, rngs=rngs, deterministic=deterministic)
                action_padding_mask_binary = (action_padding_mask == 0)[:, :, None]
                # loss_weight = (torch.arange(1, y_pred.shape[1] + 1, device=device, dtype=torch.float32) / y_pred.shape[1])[None, :, None]
                loss = jnp.mean(((y_pred - y) ** 2) * action_padding_mask_binary)
                # diff = (y_pred - y) * loss_weight
                # loss = torch.mean(torch.masked_select(diff, action_padding_mask_binary) ** 2)
                return loss


            def val_episode(var_collect, episode_num, rngs):
                episode = val_dataset.get_episode(episode_num)
                episode = jnp.array(torch.unsqueeze(episode, 0).numpy())
                batch = episode[:, :, :-1]
                history, action, y, action_padding_mask = val_dataset[episode_num:episode_num+1]
                history = jnp.array(history.numpy())
                action = jnp.array(action.numpy())
                y = jnp.array(y.numpy())
                action_padding_mask = jnp.array(action_padding_mask.numpy())
                predicted_states = apply_batch(var_collect, batch[:, history_length-1, :], history, action, y, action_padding_mask, rngs)
                return np.array(predicted_states)
                

            def val_loop(state, var_collect, val_loader, rngs):
                val_loss = 0.0
                t_val = tqdm.tqdm(val_loader)
                for i, (history, action, y, action_padding_mask) in enumerate(t_val):
                    history = jnp.array(history.numpy())
                    action = jnp.array(action.numpy())
                    y = jnp.array(y.numpy())
                    action_padding_mask = jnp.array(action_padding_mask.numpy())
                    val_loss += loss_fn(state, var_collect, history, action, y, action_padding_mask, global_rngs, True).item()
                    t_val.set_description(f'Validation Loss: {(val_loss / (i + 1)):.4f}')
                    t_val.refresh()
                val_loss /= len(val_loader)
                return val_loss

            def visualize_episode(epoch_num: int, episode_num, val_dataset, rngs):
                val_collect = model.init(init_rngs, jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask)
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                # raw_restored = orbax_checkpointer.restore(glob.glob('/home/haoru/lecar/lecar-car/model_checkpoint/*/*/')[-1])
                # print("path", os.path.join(save_model_folder_path, f"{epoch_num}"))
                raw_restored = orbax_checkpointer.restore(resume_model_folder_path)
                val_collect['params'] = raw_restored['params']
                predicted_states = val_episode(val_collect, episode_num, rngs)
                episode = val_dataset.get_episode(episode_num)

                fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                axs[0, 0].plot(episode[:, 0], episode[:, 1], label='Ground Truth', marker='o', markersize=5)
                axs[0, 0].plot(predicted_states[0, :, 0], predicted_states[0, :, 1], label='Predicted', marker='x', markersize=5)
                axs[0, 0].legend()
                axs[0, 0].axis('equal')

                predict_x = np.arange(0, predicted_states.shape[1]) + episode.shape[0] - predicted_states.shape[1]
                axs[0, 1].plot(episode[:, 3], label='Ground Truth vx')
                axs[0, 1].plot(episode[:, 4], label='Ground Truth vy')
                axs[0, 1].plot(predict_x, predicted_states[0, :, 3], label='Predicted vx')
                axs[0, 1].plot(predict_x, predicted_states[0, :, 4], label='Predicted vy')
                axs[0, 1].legend()

                axs[1, 1].plot(episode[:, 5], label='Ground Truth v_yaw')
                axs[1, 1].plot(predict_x, predicted_states[0, :, 5], label='Predicted v_yaw')
                axs[1, 1].legend()

                fig.tight_layout()
                fig.savefig(os.path.join(tmp_dir, f"train-eval-episode-{epoch_num}.png"))
                # fig.savefig('episode.png')
                plt.close(fig)

            train_losses = []
            val_losses = []
            val_epoch_nums = []

            for epoch in range(num_train_epochs):
                running_loss = 0.0
                t = tqdm.tqdm(train_loader)
                for i, (history, action, y, action_padding_mask) in enumerate(t):
                    history = jnp.array(history.numpy())
                    action = jnp.array(action.numpy())
                    y = jnp.array(y.numpy())
                    action_padding_mask = jnp.array(action_padding_mask.numpy())

                    def this_loss_fn(var_collect, history, action, y, action_padding_mask):
                        return loss_fn(global_state, var_collect, history, action, y, action_padding_mask, global_rngs)
                    
                    grad_fn = jax.value_and_grad(this_loss_fn, has_aux=False)

                    loss, grads = grad_fn(global_var, history, action, y, action_padding_mask)
                    loss_item = loss.item()
                    running_loss += loss_item

                    t.set_description(f'Epoch {epoch + 1}, Loss: {(running_loss / (i + 1)):.4f}, LR: {learning_rate_fn(global_state.step):.6f}')
                    t.refresh()

                    global_state = global_state.apply_gradients(grads=grads["params"])
                    global_var['params'] = global_state.params

                running_loss /= len(train_loader)
                train_losses.append(running_loss)
                print(save_model_folder_path)
                checkpoint_manager.save(epoch+1, global_state, save_kwargs={'save_args': save_args})
                if (epoch + 1) % val_every == 0 or epoch == 0:
                    # resume_model_checkpint = epoch + 1
                    resume_model_folder_path = os.path.join(save_model_folder_path, f"{epoch + 1}", "default")
                if (epoch + 1) % val_every == 0:
                    visualize_episode(epoch + 1, 1, val_dataset, global_rngs)
                    val_loss = val_loop(global_state, global_var, val_loader, global_rngs)
                    val_losses.append(val_loss)
                    val_epoch_nums.append(epoch + 1)
                    print(f'Validation Loss: {val_loss:.4f}')
       

            train_epoch_nums = list(range(1, num_epochs + 1))
            plt.figure()
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_epoch_nums, val_losses, label='Val Loss')
            plt.legend()
            # plt.savefig('train_val_loss.png')
            plt.savefig(os.path.join(tmp_dir, f"train_val_loss.png"))
            

            # model.eval()
            visualize_episode(epoch + 1, 1, val_dataset, global_rngs)
            test_loss = val_loop(global_state, global_var, test_loader, global_rngs)
            print(f'Test Loss: {test_loss:.4f}')

