import jax
import numpy as np
import jax.numpy as jnp
from car_dynamics.models_jax.utils import normalize_angle_tensor, fold_angle_tensor


#----------------- reward functions -----------------#

def reward_track_fn(goal_list: jnp.ndarray, defaul_speed: float):
    def reward(state, action, discount):
        """
            - state: contains current state of the car
        """
        num_rollouts = action.shape[0]
        horizon = action.shape[1]
        reward_rollout = jnp.zeros((num_rollouts))
        reward_activate = jnp.ones((num_rollouts))
        
        for h in range(horizon):
            
            state_step = state[h+1]
            action_step = action[:, h]
            dist = jnp.linalg.norm(state_step[:, :2] - goal_list[h+1, :2], axis=1)
            vel_diff = jnp.linalg.norm(state_step[:, 3:4] - defaul_speed, axis=1)
            reward = -dist - 0.0 * vel_diff - 0.0 * jnp.linalg.norm(action_step[:, 1:2], axis=1)
            # reward = - 0.4 * dist - 0.0 * jnp.norm(action_step, dim=1) - 0.0 * vel_diff - 0.1 * jnp.log(1 + dist)
            # reward = - 0.4 * dist
            reward_rollout += reward *(discount ** h) * reward_activate
        return reward_rollout
    return reward

#----------------- rollout functions -----------------#

def rollout_fn_select(model_struct, model, dt, L, LR):
    
    # @jax.jit
    def rollout_fn_dbm(obs_history, state, action, dynamic_params_tuple, debug=False):
        print(state.shape)
        assert state.shape[1] == 6
        assert action.shape[1] == 2
        LF, LR, MASS, DT, K_RFY, K_FFY, Iz, Ta, Tb, Sa, Sb, mu, Cf, Cr, Bf, Br, hcom, fr = dynamic_params_tuple
        next_state = model.step(state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4], state[:, 5], action[:, 0], action[:, 1], 
                                LF, LR, MASS, DT, K_RFY, K_FFY, Iz, Ta, Tb, Sa, Sb, mu, Cf, Cr, Bf, Br, hcom, fr)
        # next state is [x, y, psi, vx, vy, omega]
        next_state = jnp.stack(next_state, axis=1)

        return next_state, {}
    
    # @jax.jit
    def rollout_fn_tansformer_torch(key, obs_history, state, action, dynamic_params_tuple, debug=False):
        assert state.shape[1] == 6
        assert action.shape[2] == 2
        next_state = model.step(obs_history, state, action)
        next_state = np.swapaxes(next_state, 0, 1).tolist()
        return next_state, key, {}
    
    # @jax.jit
    def rollout_fn_tansformer_jax(key, obs_history, state, action, dynamic_params_tuple, debug=False):
        assert state.shape[1] == 6
        assert action.shape[2] == 2
        
        next_state = model.step(key, obs_history, state, action)
        next_state = jnp.swapaxes(next_state, 0, 1)
        
        return next_state, {}
    
    if model_struct == 'dbm':
        return rollout_fn_dbm
    elif model_struct == 'transformer-torch':
        return rollout_fn_tansformer_torch
    elif model_struct == 'transformer-jax':
        return rollout_fn_tansformer_jax
    else:
        raise Exception(f"model_struct {model_struct} not supported!")
    
    
def rollout_fn_jax(model):  
    # @jax.jit
    def rollout_fn_tansformer_jax(key, obs_history, state, action, dynamic_params_tuple, debug=False):
        assert state.shape[1] == 6
        assert action.shape[2] == 2
        
        next_state = model.step(key, obs_history, state, action)
        next_state = jnp.swapaxes(next_state, 0, 1)
        
        return next_state, {}
    
    return rollout_fn_tansformer_jax