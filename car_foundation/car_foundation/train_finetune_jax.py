from functools import partial
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax
from flax.training import orbax_utils
# from datasets import load_dataset
from flax import linen as nn
from flax.training import train_state

from car_foundation import CAR_FOUNDATION_DATA_DIR, CAR_FOUNDATION_MODEL_DIR
from car_foundation.dataset import DynamicsDataset, IssacSimDataset, MujocoDataset
from car_foundation.models import TorchMLP, TorchTransformer, TorchTransformerDecoder, TorchGPT2
from car_foundation.jax_models import JaxTransformerDecoder, JaxMLP, JaxCNN, JaxGRU, JaxLSTM
from car_foundation.utils import generate_subsequences, generate_subsequences_hf, align_yaw, align_yaw_jax

from car_dynamics.controllers_jax import MPPIController, rollout_fn_select, MPPIRunningParams, void_fn

import sys
import datetime
import os
import glob
import time
import math
import random
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import wandb
import pytorch_warmup as warmup

# np.set_printoptions(threshold=sys.maxsize)

PARAMS_KEY = "params"
DROPOUT_KEY = "dropout"
INPUT_KEY = "input_rng"

torch.manual_seed(3407)
np.random.seed(3407)
random.seed(3407)

history_length = 251
prediction_length = 50
delays = None
teacher_forcing = False

VANILLA_FINETUNE = True
NEW_FINETUNE = False
ATTACK = False
ATTN_MASK = False
REGULARIZE_FINETUNE = False
SLAM_FINETUNE = True
REG_LAMBDA = 1 

assert not (VANILLA_FINETUNE and NEW_FINETUNE) #cannot run both at the same time

if VANILLA_FINETUNE:
    lr_begin = 5e-5
    warmup_period = 2
    num_epochs = 400 #200
    load_checkpoint = True
    resume_model_checkpoint = 400
    resume_model_name = "XXX"


resume_model_folder_path = os.path.join(CAR_FOUNDATION_MODEL_DIR, resume_model_name, f"{resume_model_checkpoint}")

val_every = 20

batch_size = 1024 #8 #1024 #8 for MPPI
if SLAM_FINETUNE:
    batch_size = 128
lambda_l2 = 1e-4


aggregated_data_paths = ["DATASET_PATH",]

rehearsal_datapath = "REHEARSAL_DATASET_PATH"

comment = 'jax'

num_workers = 6

state_dim = 6
action_dim = 2
latent_dim = 64 #128
num_heads = 4
num_layers = 2
dropout = 0.1

save_model_folder_prefix = datetime.datetime.now().isoformat(timespec='milliseconds')
save_model_folder_path = os.path.join(CAR_FOUNDATION_MODEL_DIR, f'{save_model_folder_prefix}-model_checkpoint')

# architecture = 'decoder'
# architecture = 'mlp'
# architecture = 'cnn'
# architecture = 'gru'
architecture = 'decoder'

# model = TorchTransformer(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, dropout)

if architecture == 'decoder':
    model = JaxTransformerDecoder(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, dropout, history_length - 1, prediction_length, jnp.bfloat16, name=architecture)
elif architecture == 'mlp':
    model = JaxMLP([128, 128, 128, 128, 128], state_dim, 0.1, name=architecture)
elif architecture == 'cnn':
    model = JaxCNN([8, 8, 8, 8], [3, 3, 3, 3], [3, 3, 3, 3], state_dim, 0.1, name=architecture)
elif architecture == 'gru':
    model = JaxGRU(64, state_dim, 8, 0.1, name=architecture)
elif architecture == "lstm":
    # model = JaxLSTM(16, state_dim, 8, 0.1, name=architecture)
    model = JaxLSTM(32, state_dim, 0.1, name=architecture)

_key = jax.random.PRNGKey(0)

dummy_history = jnp.ones((batch_size, history_length-1, state_dim + action_dim), dtype=jnp.float32)
dummy_prediction = jnp.ones((batch_size, prediction_length, action_dim), dtype=jnp.float32)
tabulate_fn = nn.tabulate(model, _key)
print("Model Param Count", tabulate_fn(dummy_history, dummy_prediction))



# Load the dataset
binary_mask = False # type(model) == TorchGPT2


if SLAM_FINETUNE:
    slam_dataset_files = []
    vicon_dataset_files = []
    for dataset_path in aggregated_data_paths:
        files = glob.glob(os.path.join(dataset_path, '*.pkl'))
        if "slam" in dataset_path:
            slam_dataset_files = slam_dataset_files + files
        elif "vicon" in dataset_path:
            vicon_dataset_files = vicon_dataset_files + files
else:
    dataset_files = []
    for dataset_path in aggregated_data_paths:
        files = glob.glob(os.path.join(dataset_path, '*.pkl'))
        
        #balance out the assetto corsa dataset a bit
        if "dallara" in dataset_path:
            num_to_remove = 33000
        elif "bmw" in dataset_path:
            num_to_remove = 15000
        elif "miata" in dataset_path:
            num_to_remove = 0
        else:
            num_to_remove = 0
            
        random.shuffle(files)

        dataset_files = dataset_files + files[num_to_remove:]

if SLAM_FINETUNE:
    total_len = len(slam_dataset_files)
    split_70 = int(total_len * 0.7)
    split_20 = int(total_len * 0.9)
    data_70 = slam_dataset_files[:split_70]
    data_20 = slam_dataset_files[split_70:split_20]
    data_10 = slam_dataset_files[split_20:]
    train_dataset = MujocoDataset(data_70, history_length, prediction_length, delays=delays, teacher_forcing=teacher_forcing, binary_mask=binary_mask,attack=False, filter=False)
    vicon_train_dataset = MujocoDataset(vicon_dataset_files[:split_70], history_length, prediction_length, delays=delays, teacher_forcing=teacher_forcing, binary_mask=binary_mask,attack=False, filter=False)
    train_dataset.y = vicon_train_dataset.y

    val_dataset = MujocoDataset(data_20, history_length, prediction_length, delays=delays, mean=train_dataset.mean, teacher_forcing=teacher_forcing, std=train_dataset.std, binary_mask=binary_mask, attack=False)
    vicon_val_dataset = MujocoDataset(vicon_dataset_files[split_70:split_20], history_length, prediction_length, delays=delays, mean=train_dataset.mean, teacher_forcing=teacher_forcing, std=train_dataset.std, binary_mask=binary_mask, attack=False)
    val_dataset.y = vicon_val_dataset.y

    test_dataset = MujocoDataset(data_10, history_length, prediction_length, delays=delays, mean=train_dataset.mean, teacher_forcing=teacher_forcing, std=train_dataset.std, binary_mask=binary_mask, attack=False)
    vicon_test_dataset = MujocoDataset(vicon_dataset_files[split_20:], history_length, prediction_length, delays=delays, mean=train_dataset.mean, teacher_forcing=teacher_forcing, std=train_dataset.std, binary_mask=binary_mask, attack=False)
    test_dataset.y = vicon_test_dataset.y

    rehearsal_dataset = MujocoDataset(rehearsal_datapath, history_length, prediction_length, delays=delays, mean=train_dataset.mean, teacher_forcing=teacher_forcing, std=train_dataset.std, binary_mask=binary_mask, attack=False)
else:
    random.shuffle(dataset_files)
    total_len = len(dataset_files)
    split_70 = int(total_len * 0.7)
    split_20 = int(total_len * 0.9)
    data_70 = dataset_files[:split_70]
    data_20 = dataset_files[split_70:split_20]
    data_10 = dataset_files[split_20:]

    train_dataset = MujocoDataset(data_70, history_length, prediction_length, delays=delays, teacher_forcing=teacher_forcing, binary_mask=binary_mask,attack=ATTACK, filter=False)
    print("train data length", len(train_dataset))

    val_dataset = MujocoDataset(data_20, history_length, prediction_length, delays=delays, mean=train_dataset.mean, teacher_forcing=teacher_forcing, std=train_dataset.std, binary_mask=binary_mask, attack=ATTACK)
    test_dataset = MujocoDataset(data_10, history_length, prediction_length, delays=delays, mean=train_dataset.mean, teacher_forcing=teacher_forcing, std=train_dataset.std, binary_mask=binary_mask, attack=ATTACK)



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

if SLAM_FINETUNE:
    rehearsal_loader = DataLoader(rehearsal_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

num_steps_per_epoch = len(train_loader)

# note = input("Please enter a note for this run: ")
note="tune"
wandb.init(
    # set the wandb project where this run will be logged
    project="transformer-sequence-prediction-ablation",
    name=note + "-" + architecture + "-" + save_model_folder_prefix,

    # track hyperparameters and run metadata
    config={
        "history_length": history_length,
        "prediction_length": prediction_length,
        "delays": delays,
        "teacher_forcing": teacher_forcing,

        "learning_rate": lr_begin,
        "warmup_period": warmup_period,
        "architecture": architecture,
        "dataset": "even_dist_data",
        "epochs": num_epochs,
        "batch_size": batch_size,
        "lambda_l2": lambda_l2,
        "dataset_path": dataset_path.split('/')[-1],
        "comment": comment,

        "state_dim": state_dim,
        "action_dim": action_dim,
        "latent_dim": latent_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dropout": dropout,
        "implementation": "jax",
        "model_path": save_model_folder_path,
        "resume": load_checkpoint,
        "resume_checkpoint_path": resume_model_folder_path,
        "resume_checkpoint": resume_model_checkpoint,
        "attack": ATTACK,
        "is_finetune": VANILLA_FINETUNE,
    }
)
print(wandb.config)
# print(f"total params: {sum(p.numel() for p in model.parameters())}")

rng = jax.random.PRNGKey(3407)
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
    decay_fn = optax.exponential_decay(lr_begin, decay_rate=0.99, transition_steps=num_steps_per_epoch, staircase=True) #decay_rate=0.993 for 800 epochs
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
options = orbax.checkpoint.CheckpointManagerOptions(create=True, save_interval_steps=val_every)
checkpoint_manager = orbax.checkpoint.CheckpointManager(save_model_folder_path, orbax_checkpointer, options)

if VANILLA_FINETUNE:
    restored = checkpoint_manager.restore(resume_model_folder_path)
    global_var['params'] = restored['model']['params']
    global_state = train_state.TrainState.create(
            apply_fn=model.apply, params=global_var[PARAMS_KEY], tx=tx
    )
    input_mean = jnp.array(restored['input_mean'])
    input_std = jnp.array(restored['input_std'])

    new_input_mean = jnp.array(train_dataset.mean, dtype=jnp.float32)
    new_input_std = jnp.array(train_dataset.std, dtype=jnp.float32)
    
    input_mean = (new_input_mean + input_mean)/2
    input_std = (new_input_std + input_std)/2
    
elif NEW_FINETUNE:
    assert NotImplementedError
else:
    input_mean = jnp.array(train_dataset.mean, dtype=jnp.float32)
    input_std = jnp.array(train_dataset.std, dtype=jnp.float32)
    
print("mean: ", input_mean.tolist())
print("std: ", input_std.tolist())


def apply_batch(var_collect, last_state, history, action, y, action_padding_mask, rngs):
    history = history.at[:, :, :6].set((history[:, :, :6] - input_mean) / input_std)
    y = y.at[:, :, :6].set((y[:, :, :6] - input_mean) / input_std)

    x = history[:, 1:, :]
    # tgt_mask = nn.Transformer.generate_square_subsequent_mask(action.size(1), device=action.device)
    if not NEW_FINETUNE:
        y_pred = model.apply(var_collect, x, action, action_padding_mask=action_padding_mask, rngs=rngs, deterministic=True) * input_std + input_mean
    else:
        assert NotImplementedError
    
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



# @partial(jax.jit, static_argnums=(7,))
def loss_fn(state, var_collect, history, action, y, action_padding_mask, rngs, deterministic=False):
    history = history.at[:, :, :6].set((history[:, :, :6] - input_mean) / input_std)
    y = y.at[:, :, :6].set((y[:, :, :6] - input_mean) / input_std)
    history = jax.lax.stop_gradient(history[:, 1:, :]) #make hustory len 250
    y = jax.lax.stop_gradient(y)
    action = jax.lax.stop_gradient(action)

    # Boolean tensor used to mask out the attention softmax input.
    #         :attr:`True` means mask out the corresponding values.

    if ATTN_MASK:
        pred_length = prediction_length
        hist_length = (history_length - 1) * 2 - 1 # length of history after positional encoding

        mask = jnp.zeros((pred_length, hist_length), dtype=jnp.bool)
        mask = jnp.tile(mask, (history.shape[0], num_heads, 1, 1))
        
        # Calculate the number of rows to be set as 1
        rng, new_rng = jax.random.split(rngs[PARAMS_KEY])

        MASK_NUM = int(jax.random.randint(new_rng, (), hist_length // 16, hist_length // 4).item())

        rng, new_rng = jax.random.split(new_rng)

        rngs[PARAMS_KEY] = new_rng

        random_columns = jax.random.randint(new_rng, (history.shape[0], num_heads, MASK_NUM), 0, hist_length)
        
        # Set the selected rows to all 1s
        batch_indices = jnp.arange(history.shape[0])[:, None, None]
        head_indices = jnp.arange(num_heads)[None, :, None]
        mask = mask.at[batch_indices, head_indices, :, random_columns].set(True)

        tgt_mask = mask
    else:
        tgt_mask = None

    y_pred = state.apply_fn(var_collect, history, action, history_padding_mask=None, action_padding_mask=action_padding_mask, tgt_mask=tgt_mask, rngs=rngs, deterministic=deterministic)
    action_padding_mask_binary = (action_padding_mask == 0)[:, :, None]
    # loss_weight = (torch.arange(1, y_pred.shape[1] + 1, device=device, dtype=torch.float32) / y_pred.shape[1])[None, :, None]
    loss = jnp.mean(((y_pred - y) ** 2) * action_padding_mask_binary)
    # diff = (y_pred - y) * loss_weight
    # loss = torch.mean(torch.masked_select(diff, action_padding_mask_binary) ** 2)
    return loss

# @partial(jax.jit, static_argnums=(7,8))
def new_loss_fn(state, var_collect, history, action, y, raw_data, action_padding_mask, rngs, deterministic=False, mppi_running_params=None):
    # state history normalization
    history = history.at[:, :, :6].set((history[:, :, :6] - input_mean) / input_std)
    # goal normalization
    y = y.at[:, :, :6].set((y[:, :, :6] - input_mean) / input_std)

    history = history[:, 1:, :]
    # history = jax.lax.stop_gradient(history[:, 1:, :])
    # y = jax.lax.stop_gradient(y)
    # action = jax.lax.stop_gradient(action)
    # raw_data = jax.lax.stop_gradient(raw_data)
    
    loss = 0.0

    #TODO can vectorize this later
    for rollout in range(history.shape[0]):
        state_history = history[rollout, :, :]
        target_actions = action[rollout, :, :]
        target_states = y[rollout, :, :]
        curr_state = raw_data[rollout, history_length - 1, :] #get last state in history
        #add last state history to target states
        target_states = jnp.vstack((state_history[-1, :6], target_states))

        # run MPPI with dynamics to get nominal actions (best actions from MPPI)
            #MPPI Inputs
            # current history + state  (should not need to feed history since it already has full history)
                # (history) nned to update running_params.state_hist
                # (state) (not even used lol)
            # a_curr -> a_horizon (action)
            # x_curr -> x_horizon (y)
            # #a_mean (optimal actions)
            
        mppi_running_params = MPPIRunningParams(
                    a_mean = target_actions,
                    a_cov = mppi_running_params.a_cov,
                    prev_a = mppi_running_params.prev_a, #not sure if this is needed
                    state_hist = state_history,
                    key = mppi_running_params.key,
                )

        target_pos_tensor = target_states
        mppi_action, mppi_running_params, mppi_info = mppi(curr_state, target_pos_tensor, mppi_running_params, dynamic_params_tuple, vis_optim_traj=True, nn_state=state, nn_var=var_collect)
        pred_actions = mppi_info["a_mean_jnp"]
        # calculate loss with actual actions taken
        loss += jnp.mean((pred_actions - target_actions) ** 2)
    # return the loss
  
    return loss/history.shape[0]

def val_episode(var_collect, episode_num, rngs):
    episode = val_dataset.get_episode(episode_num)
    episode = jnp.array(torch.unsqueeze(episode, 0).numpy())
    batch = episode[:, :, :-1]
    history, action, y, action_padding_mask, raw_data = val_dataset[episode_num:episode_num+1]
    history = jnp.array(history.numpy())
    action = jnp.array(action.numpy())
    y = jnp.array(y.numpy())
    action_padding_mask = jnp.array(action_padding_mask.numpy())
    predicted_states = apply_batch(var_collect, batch[:, history_length-1, :], history, action, y, action_padding_mask, rngs)
    return np.array(predicted_states)
    
def val_loop(state, var_collect, val_loader, rngs):
    val_loss = 0.0
    t_val = tqdm.tqdm(val_loader)
    for i, (history, action, y, action_padding_mask, raw_data) in enumerate(t_val):
        history = jnp.array(history.numpy())
        action = jnp.array(action.numpy())
        y = jnp.array(y.numpy())
        raw_data = jnp.array(raw_data.numpy()) 
        action_padding_mask = jnp.array(action_padding_mask.numpy())
        if not NEW_FINETUNE:
            val_loss += loss_fn(state, var_collect, history, action, y, action_padding_mask, global_rngs, True).item()
        else:
            assert NotImplementedError
            #for comparison with vanilla finetune
            # val_loss += loss_fn(state, var_collect, history, action, y, action_padding_mask, global_rngs, True).item()
            # val_loss += new_loss_fn(state, var_collect, history, action, y, raw_data, action_padding_mask, global_rngs, mppi_running_params=mppi_running_params).item()
        t_val.set_description(f'Validation Loss: {(val_loss / (i + 1)):.4f}')
        t_val.refresh()
    val_loss /= len(val_loader)
    return val_loss

def visualize_episode(epoch_num: int, episode_num, val_dataset, rngs):
    val_collect = model.init(init_rngs, jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    raw_restored = orbax_checkpointer.restore(os.path.join(save_model_folder_path, f"{epoch_num}", "default"))
    # import ipdb; ipdb.set_trace()
    val_collect['params'] = raw_restored['model']['params']
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
    fig.savefig('episode.png')
    plt.close(fig)
    wandb.log({"episode": wandb.Image('episode.png')})

train_losses = []
val_losses = []
val_epoch_nums = []

for epoch in range(num_epochs):
    running_loss = 0.0
    t = tqdm.tqdm(train_loader)
    for i, (history, action, y, action_padding_mask, raw_data) in enumerate(t):
        if SLAM_FINETUNE:
            (rehearsal_history, rehearsal_action, rehearsal_y, rehearsal_action_padding_mask, rehearsal_raw_data) = next(iter(rehearsal_loader))
            history = np.concatenate((history.numpy(), rehearsal_history.numpy()), axis=0)
            action = np.concatenate((action.numpy(), rehearsal_action.numpy()), axis=0)
            y = np.concatenate((y.numpy(), rehearsal_y.numpy()), axis=0)
            action_padding_mask = np.concatenate((action_padding_mask.numpy(), rehearsal_action_padding_mask.numpy()), axis=0)
            raw_data = np.concatenate((raw_data.numpy(), rehearsal_raw_data.numpy()), axis=0)
            
            permutation = np.random.permutation(history.shape[0])
            
            history = jnp.array(history[permutation])
            action = jnp.array(action[permutation])
            y = jnp.array(y[permutation])
            action_padding_mask = jnp.array(action_padding_mask[permutation])
            raw_data = jnp.array(raw_data[permutation])

        else:
            history = jnp.array(history.numpy()) # [batch_size, history_length, state_dim + action_dim]
            action = jnp.array(action.numpy()) #   [batch_size, prediction_length, action_dim]
            y = jnp.array(y.numpy()) #           [batch_size, prediction_length, state_dim]
            raw_data = jnp.array(raw_data.numpy())# [batch_size, history_length + prediction_length, state_dim + action_dim]
            
            action_padding_mask = jnp.array(action_padding_mask.numpy())

        def this_loss_fn(var_collect, history, action, y, action_padding_mask):
            if not NEW_FINETUNE:
                return loss_fn(global_state, var_collect, history, action, y, action_padding_mask, global_rngs)
            elif NEW_FINETUNE and REGULARIZE_FINETUNE:
                assert NotImplementedError
                # return (new_loss_fn(dynamics.state, var_collect, history, action, y, raw_data, action_padding_mask, global_rngs, mppi_running_params=mppi_running_params) +
                #     REG_LAMBDA * loss_fn(dynamics.state, var_collect, history, action, y, action_padding_mask, global_rngs))
            else:
                assert NotImplementedError
                # return new_loss_fn(dynamics.state, var_collect, history, action, y, raw_data, action_padding_mask, global_rngs, mppi_running_params=mppi_running_params)
        
        grad_fn = jax.value_and_grad(this_loss_fn, has_aux=False)

        # forward and backward
        if not NEW_FINETUNE:
            loss, grads = grad_fn(global_var, history, action, y, action_padding_mask)
        else:
            assert NotImplementedError
            # loss, grads = grad_fn(dynamics.var, history, action, y, action_padding_mask)
        loss_item = loss.item()
        running_loss += loss_item

        # print(f"Gradients: {grads}")

        # if loss_item > 3.0:
        #     episode_nums = batch[:, -1, -1].cpu().numpy().astype(int)
        #     if problematic_episodes.isdisjoint(episode_nums):
        #         problematic_episodes.update(episode_nums)
        #     else:
        #         problematic_episodes = problematic_episodes.intersection(episode_nums)    
        #     print(f'Problematic episodes: {problematic_episodes}')
        if not NEW_FINETUNE:
            t.set_description(f'Epoch {epoch + 1}, Loss: {(running_loss / (i + 1)):.4f}, LR: {learning_rate_fn(global_state.step):.6f}')
        else:
            assert NotImplementedError
            # t.set_description(f'Epoch {epoch + 1}, Loss: {(running_loss / (i + 1)):.4f}, LR: {learning_rate_fn(dynamics.state.step):.6f}')
        t.refresh()

        # apply gradients
        if not NEW_FINETUNE:
            global_state = global_state.apply_gradients(grads=grads["params"])
            global_var['params'] = global_state.params
        else:
            assert NotImplementedError
            # dynamics.state = dynamics.state.apply_gradients(grads=grads["params"])
            # dynamics.var["params"] = dynamics.state.params

    running_loss /= len(train_loader)
    train_losses.append(running_loss)
    if not NEW_FINETUNE:
        wandb.log({"train_loss": running_loss, "learning_rate": learning_rate_fn(global_state.step)})
    else:
        assert NotImplementedError
        # wandb.log({"train_loss": running_loss, "learning_rate": learning_rate_fn(dynamics.state.step)})
    print(save_model_folder_path)
    # import ipdb; ipdb.set_trace()
    if not NEW_FINETUNE:
        ckpt = {'model': global_state, 'input_mean': input_mean, 'input_std': input_std}
    else:
        assert NotImplementedError
        # ckpt = {'model': dynamics.state, 'input_mean': input_mean, 'input_std': input_std}
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpoint_manager.save(epoch+1, ckpt, save_kwargs={'save_args': save_args})

    if (epoch + 1) % val_every == 0:
        visualize_episode(epoch + 1, 1, val_dataset, global_rngs)
        if not NEW_FINETUNE:
            val_loss = val_loop(global_state, global_var, val_loader, global_rngs)
        else:
            assert NotImplementedError
            # val_loss = val_loop(dynamics.state, dynamics.var, val_loader, global_rngs)
        val_losses.append(val_loss)
        val_epoch_nums.append(epoch + 1)
        print(f'Validation Loss: {val_loss:.4f}')
        wandb.log({"val_loss": val_loss})
    # torch.save(model.state_dict(), f'model_checkpoint.pth')
    # torch.save(optm.state_dict(), f'optm_checkpoint.pth')
    # torch.save(scheduler.state_dict(), f'scheduler_checkpoint.pth')
    # torch.save(warmup_scheduler.state_dict(), f'warmup_scheduler_checkpoint.pth')
    # wandb.save('model_checkpoint.pth')
    # wandb.save('optm_checkpoint.pth')
    # wandb.save('scheduler_checkpoint.pth')

train_epoch_nums = list(range(1, num_epochs + 1))
plt.figure()
plt.plot(train_epoch_nums, train_losses, label='Train Loss')
plt.plot(val_epoch_nums, val_losses, label='Val Loss')
plt.legend()
plt.savefig('train_val_loss.png')
# plt.show()

# model.eval()
visualize_episode(epoch + 1, 1, val_dataset, global_rngs)
if not NEW_FINETUNE:
    test_loss = val_loop(global_state, global_var, test_loader, global_rngs)
else:
    assert NotImplementedError
    # test_loss = val_loop(dynamics.state, dynamics.var, test_loader, global_rngs)
print(f'Test Loss: {test_loss:.4f}')

# Save the model
# torch.save(model.state_dict(), 'model.pth')
wandb.save('model_checkpoint/')

wandb.finish()