
import optax
import flax
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import torch
from car_foundation.utils import generate_subsequences, generate_subsequences_hf, align_yaw, align_yaw_jax
import orbax
import matplotlib.pyplot as plt
import os
import wandb


def create_learning_rate_fn(warmup_period, lr_begin, num_steps_per_epoch):
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=lr_begin, transition_steps=warmup_period)
    decay_fn = optax.exponential_decay(lr_begin, decay_rate=0.99, transition_steps=num_steps_per_epoch, staircase=True) #decay_rate=0.993 for 800 epochs
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[warmup_period]
    )
    return schedule_fn

def visualize_episode(epoch_num: int, episode_num, val_dataset, rngs, model, init_rngs, save_model_folder_path, jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask, wandb_run, history_length, input_mean, input_std):
    val_collect = model.init(init_rngs, jax_history_input, jax_prediction_input, jax_history_mask, jax_prediction_mask)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    raw_restored = orbax_checkpointer.restore(os.path.join(save_model_folder_path, f"{epoch_num}", "default"))
    # import ipdb; ipdb.set_trace()
    val_collect['params'] = raw_restored['model']['params']
    predicted_states = val_episode(val_collect, episode_num, rngs, val_dataset, history_length, input_mean, input_std, model)
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
    wandb_run.log({"episode": wandb.Image('episode.png')})
    
    
def val_episode(var_collect, episode_num, rngs, val_dataset, history_length, input_mean, input_std, model):
    episode = val_dataset.get_episode(episode_num)
    episode = jnp.array(torch.unsqueeze(episode, 0).numpy())
    batch = episode[:, :, :-1]
    history, action, y, action_padding_mask, raw_data = val_dataset[episode_num:episode_num+1]
    history = jnp.array(history.numpy())
    action = jnp.array(action.numpy())
    y = jnp.array(y.numpy())
    action_padding_mask = jnp.array(action_padding_mask.numpy())
    predicted_states = apply_batch(var_collect, batch[:, history_length-1, :], history, action, y, action_padding_mask, rngs, input_mean, input_std, model)
    return np.array(predicted_states)
    
def val_loop(state, var_collect, val_loader, rngs, global_rngs, input_mean, input_std, prediction_length, ATTN_MASK, history_length, num_heads, PARAMS_KEY):
    val_loss = 0.0
    t_val = tqdm.tqdm(val_loader)
    for i, (history, action, y, action_padding_mask, raw_data) in enumerate(t_val):
        history = jnp.array(history.numpy())
        action = jnp.array(action.numpy())
        y = jnp.array(y.numpy())
        raw_data = jnp.array(raw_data.numpy()) 
        action_padding_mask = jnp.array(action_padding_mask.numpy())
        val_loss += loss_fn(state, var_collect, history, action, y, action_padding_mask, global_rngs, input_mean, input_std, prediction_length, ATTN_MASK, history_length, num_heads, PARAMS_KEY, True).item()
        t_val.set_description(f'Validation Loss: {(val_loss / (i + 1)):.4f}')
        t_val.refresh()
    val_loss /= len(val_loader)
    return val_loss


# @partial(jax.jit, static_argnums=(7,))
def loss_fn(state, var_collect, history, action, y, action_padding_mask, rngs, input_mean, input_std, prediction_length, ATTN_MASK, history_length, num_heads, PARAMS_KEY, deterministic=False):
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

    y_pred, _ = state.apply_fn(var_collect, history, action, history_padding_mask=None, action_padding_mask=action_padding_mask, tgt_mask=tgt_mask, rngs=rngs, deterministic=deterministic)
    action_padding_mask_binary = (action_padding_mask == 0)[:, :, None]
    # loss_weight = (torch.arange(1, y_pred.shape[1] + 1, device=device, dtype=torch.float32) / y_pred.shape[1])[None, :, None]
    loss = jnp.mean(((y_pred - y) ** 2) * action_padding_mask_binary)
    # diff = (y_pred - y) * loss_weight
    # loss = torch.mean(torch.masked_select(diff, action_padding_mask_binary) ** 2)
    return loss

def apply_batch(var_collect, 
                last_state, 
                history, 
                action, 
                y, 
                action_padding_mask, 
                rngs, 
                input_mean, 
                input_std, 
                model
):
    history = history.at[:, :, :6].set((history[:, :, :6] - input_mean) / input_std)
    y = y.at[:, :, :6].set((y[:, :, :6] - input_mean) / input_std)

    x = history[:, 1:, :]
    # tgt_mask = nn.Transformer.generate_square_subsequent_mask(action.size(1), device=action.device)
    y_pred, attn_weights = model.apply(var_collect, x, action, action_padding_mask=action_padding_mask, rngs=rngs, deterministic=True)
    y_pred = y_pred * input_std + input_mean
    print(x.shape)
    print(attn_weights.shape)
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


