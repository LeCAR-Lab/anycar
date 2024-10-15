from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
flags.DEFINE_string('architecture', "", 'decoder / cnn / cnn-medium / cnn-small / gru / gru-medium / gru-small / lstm / lstm-medium / lstm-small')
flags.DEFINE_string('data_scale', "", "1M / 10M / 100M")
flags.DEFINE_bool('attack', False, "Whether to attack the model")
flags.DEFINE_bool('mask_out', True, "Whether to use attention mask")
flags.DEFINE_bool("add_noise", False, "Whether to add noise to the input data")

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
from car_foundation.train_utils import create_learning_rate_fn, loss_fn, val_episode, val_loop, visualize_episode
# new imports
# from car_ros2.utils import load_dynamic_params, load_mppi_params, load_env_params_mujoco, load_env_params_numeric, load_env_params_isaacsim, load_env_params_unity, load_env_params_assettocorsa
# from car_dynamics.models_jax import TuneDynamicsJax
# from car_dynamics.controllers_jax import MPPIController, rollout_fn_select, MPPIRunningParams, void_fn

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







def main(argv):
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

    VANILLA_FINETUNE = False
    NEW_FINETUNE = False
    
    ATTACK = FLAGS.attack
    ATTN_MASK = FLAGS.mask_out
    ADD_NOISE = FLAGS.add_noise
    
    REGULARIZE_FINETUNE = False
    SLAM_FINETUNE = False
    REG_LAMBDA = 1 

    assert not (VANILLA_FINETUNE and NEW_FINETUNE) #cannot run both at the same time
    lr_begin = 5e-4 #5e-4
    warmup_period = 500 #500
    
    num_epochs = 400 #800
    val_every = 5
    load_checkpoint = False

    resume_model_checkpoint = 0
    resume_model_name = ""

    resume_model_folder_path = os.path.join(CAR_FOUNDATION_MODEL_DIR, resume_model_name, f"{resume_model_checkpoint}")

    batch_size = 1024 #8 #1024 #8 for MPPI

    lambda_l2 = 1e-4
    data_scale = FLAGS.data_scale
    if data_scale == "1M":
        aggregated_data_paths = ['XXX']
    elif data_scale == "10M":
        aggregated_data_paths = ['XXX']
    elif data_scale == "100M":
        aggregated_data_paths = ['XXX']
    else:
        raise ValueError("Invalid data scale")
    
    aggregated_data_paths_eval = ['XXX']


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

    architecture = FLAGS.architecture
    print("Architecture: ", architecture)
    if architecture == 'decoder':
        model = JaxTransformerDecoder(state_dim, action_dim, state_dim, latent_dim, num_heads, num_layers, dropout, history_length - 1, prediction_length, jnp.bfloat16, name=architecture)
    elif architecture == 'mlp':
        model = JaxMLP([128, 128, 128, 128, 128], state_dim, 0.1, name=architecture)
    elif architecture == 'mlp-medium':
        model = JaxMLP([64, 64, 64, 64], state_dim, 0.1, name=architecture)
    elif architecture == 'mlp-small':
        model = JaxMLP([32, 32, 32, 32], state_dim, 0.1, name=architecture)
    elif architecture == 'cnn':
        model = JaxCNN([8, 8, 8, 8], [3, 3, 3, 3], [3, 3, 3, 3], state_dim, 0.1, name=architecture)
    elif architecture == 'cnn-medium':
        model = JaxCNN([8, 8, 8], [3, 3, 3], [12, 12, 12], state_dim, 0.1, name=architecture)
    elif architecture == 'cnn-small':
        model = JaxCNN([8, 4, 4], [3, 3, 3], [24, 24, 24], state_dim, 0.1, name=architecture)
    # elif architecture == 'cnn-large':
    #     model = JaxCNN([256, 128, 64, 32], [5, 5, 5, 5], [3, 3, 3, 3], state_dim, 0.1, name=architecture)
    elif architecture == 'gru':
        model = JaxGRU(64, state_dim, 8, 0.1, name=architecture)
    elif architecture == 'gru-medium':
        model = JaxGRU(16, state_dim, 4, 0.1, name=architecture)
    elif architecture == 'gru-small':
        model = JaxGRU(8, state_dim, 8, 0.1, name=architecture)
    elif architecture == "lstm":
        # model = JaxLSTM(16, state_dim, 8, 0.1, name=architecture)
        model = JaxLSTM(32, state_dim, 0.1, name=architecture)
    elif architecture == "lstm-medium":
        model = JaxLSTM(16, state_dim, 0.1, name=architecture)
    elif architecture == "lstm-small":
        model = JaxLSTM(8, state_dim, 0.1, name=architecture)
    else:
        raise ValueError("Invalid architecture")

    _key = jax.random.PRNGKey(0)

    dummy_history = jnp.ones((batch_size, history_length-1, state_dim + action_dim), dtype=jnp.float32)
    dummy_prediction = jnp.ones((batch_size, prediction_length, action_dim), dtype=jnp.float32)
    tabulate_fn = nn.tabulate(model, _key)
    print("Model Param Count", tabulate_fn(dummy_history, dummy_prediction))


    # Load the dataset
    binary_mask = False # type(model) == TorchGPT2


    dataset_files = []
    for dataset_path in aggregated_data_paths:
        files = glob.glob(os.path.join(dataset_path, '*.pkl'))
        num_to_remove = 0
            
        random.shuffle(files)

        dataset_files = dataset_files + files[num_to_remove:]

    dataset_files_eval = []
    for dataset_path in aggregated_data_paths_eval:
        files = glob.glob(os.path.join(dataset_path, '*.pkl'))
        random.shuffle(files)
        dataset_files_eval = dataset_files_eval + files

    random.shuffle(dataset_files)
    # total_len = len(dataset_files)
    # split_70 = int(total_len * 0.7)
    # split_20 = int(total_len * 0.9)
    data_70 = dataset_files
    data_20 = dataset_files_eval
    data_10 = dataset_files_eval

    train_dataset = MujocoDataset(data_70, history_length, prediction_length, delays=delays, teacher_forcing=teacher_forcing, binary_mask=binary_mask,attack=ATTACK, filter=False, add_noise=ADD_NOISE)
    print("train data length", len(train_dataset))

    val_dataset = MujocoDataset(data_20, history_length, prediction_length, delays=delays, mean=train_dataset.mean, teacher_forcing=teacher_forcing, std=train_dataset.std, binary_mask=binary_mask, attack=ATTACK)
    test_dataset = MujocoDataset(data_10, history_length, prediction_length, delays=delays, mean=train_dataset.mean, teacher_forcing=teacher_forcing, std=train_dataset.std, binary_mask=binary_mask, attack=ATTACK)



    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    num_steps_per_epoch = len(train_loader)

    # note = input("Please enter a note for this run: ")

    run=wandb.init(
        # set the wandb project where this run will be logged
        project="transformer-sequence-prediction-ablation",
        name= architecture + "-" + save_model_folder_prefix,

        # track hyperparameters and run metadata
        config={
            "history_length": history_length,
            "prediction_length": prediction_length,
            "delays": delays,
            "teacher_forcing": teacher_forcing,
            "architecture": architecture,
            "data_scale": data_scale,

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
            "add_noise": ADD_NOISE,
            "mask_out": ATTN_MASK,
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


    learning_rate_fn = create_learning_rate_fn(warmup_period, lr_begin, num_steps_per_epoch)

    global_var = model.init(init_rngs, jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask)
    tx = optax.adamw(learning_rate_fn, weight_decay=lambda_l2)
    global_state = train_state.TrainState.create(
                apply_fn=model.apply, params=global_var[PARAMS_KEY], tx=tx
            )

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(create=True, save_interval_steps=val_every)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(save_model_folder_path, orbax_checkpointer, options)


    input_mean = jnp.array(train_dataset.mean, dtype=jnp.float32)
    input_std = jnp.array(train_dataset.std, dtype=jnp.float32)
        
    print("mean: ", input_mean.tolist())
    print("std: ", input_std.tolist())
    train_losses = []
    val_losses = []
    val_epoch_nums = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        t = tqdm.tqdm(train_loader)
        for i, (history, action, y, action_padding_mask, raw_data) in enumerate(t):
            history = jnp.array(history.numpy()) # [batch_size, history_length, state_dim + action_dim]
            action = jnp.array(action.numpy()) #   [batch_size, prediction_length, action_dim]
            y = jnp.array(y.numpy()) #           [batch_size, prediction_length, state_dim]
            raw_data = jnp.array(raw_data.numpy())# [batch_size, history_length + prediction_length, state_dim + action_dim]
            
            action_padding_mask = jnp.array(action_padding_mask.numpy())

            def this_loss_fn(var_collect, history, action, y, action_padding_mask):
                return loss_fn(global_state, var_collect, history, action, y, action_padding_mask, global_rngs, input_mean, input_std, prediction_length, ATTN_MASK, history_length, num_heads, PARAMS_KEY)
            
            grad_fn = jax.value_and_grad(this_loss_fn, has_aux=False)

            # forward and backward
            loss, grads = grad_fn(global_var, history, action, y, action_padding_mask)
        
            loss_item = loss.item()
            running_loss += loss_item

            t.set_description(f'Epoch {epoch + 1}, Loss: {(running_loss / (i + 1)):.4f}, LR: {learning_rate_fn(global_state.step):.6f}')
            
            t.refresh()

            global_state = global_state.apply_gradients(grads=grads["params"])
            global_var['params'] = global_state.params
    

        running_loss /= len(train_loader)
        train_losses.append(running_loss)
        run.log({"train_loss": running_loss, "learning_rate": learning_rate_fn(global_state.step)})
        
        print(save_model_folder_path)
        # import ipdb; ipdb.set_trace()
        if not NEW_FINETUNE:
            ckpt = {'model': global_state, 'input_mean': input_mean, 'input_std': input_std}
        
        save_args = orbax_utils.save_args_from_target(ckpt)
        checkpoint_manager.save(epoch+1, ckpt, save_kwargs={'save_args': save_args})

        if (epoch + 1) % val_every == 0:
            visualize_episode(epoch + 1, 1, val_dataset, global_rngs, model, init_rngs, save_model_folder_path, jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask, run, history_length, input_mean, input_std)
            val_loss = val_loop(global_state, global_var, val_loader, global_rngs, global_rngs,input_mean, input_std, prediction_length, ATTN_MASK, history_length, num_heads, PARAMS_KEY)
            
            val_losses.append(val_loss)
            val_epoch_nums.append(epoch + 1)
            print(f'Validation Loss: {val_loss:.4f}')
            run.log({"val_loss": val_loss})

    train_epoch_nums = list(range(1, num_epochs + 1))
    plt.figure()
    plt.plot(train_epoch_nums, train_losses, label='Train Loss')
    plt.plot(val_epoch_nums, val_losses, label='Val Loss')
    plt.legend()
    plt.savefig('train_val_loss.png')
    # plt.show()

    # model.eval()
    visualize_episode(epoch + 1, 1, val_dataset, global_rngs, model, init_rngs, save_model_folder_path, jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask, run, history_length, input_mean, input_std)
    test_loss = val_loop(global_state, global_var, test_loader, global_rngs, global_rngs, input_mean, input_std, prediction_length, ATTN_MASK, history_length, num_heads, PARAMS_KEY)

    print(f'Test Loss: {test_loss:.4f}')

    # Save the model
    # torch.save(model.state_dict(), 'model.pth')
    run.save('model_checkpoint/')

    wandb.finish()
    
if __name__ == '__main__':
  app.run(main)