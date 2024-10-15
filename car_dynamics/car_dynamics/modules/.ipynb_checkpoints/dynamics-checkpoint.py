from .models import BaseEnc1, BaseModel1
from termcolor import colored
from .pipelines import BaseModelLearning
import jax
import ray
import jax.numpy as jnp
from jax import random
import numpy as np
import torch
import itertools
from rich.progress import track
from .data_utils import normalize_dataset, save_dataset
from .networks import AffineControlNN, AffineInference,AffineControlEncNN,AffineEncInference
import optax
from flax.training import orbax_utils
import orbax.checkpoint
# from flax.training import checkpoints
from flax.training import train_state
from typing import Dict 
import flax
from copy import deepcopy
from .utils import create_missing_folder
import os

@jax.jit
def eval_step(state, inputs, labels):
    outputs = state.apply_fn({'params': state.params}, inputs)
    relative_error = jnp.abs((outputs - labels) / (labels + 1e-8))
    return jnp.mean(relative_error, axis=0)



class VanillaDynamics(BaseModel1):
    ''' Vanilla Dynamics, no shared encoder'''

    def __init__(self, 
                 input_dims,
                 output_dims,
                 action_dims,
                 key1,
                 batch_size=1024,
                 learning_rate=1e-3, 
                 verbose = 0,
                 DEBUG = False,
                 model_dir = './', 
                 env_dims = 0,
                 ):
        '''The env params are thrown out.
        '''

        super().__init__(encoder=None, 
                         key1=key1,
                         base_network=AffineControlNN, 
                         inference_network=AffineInference,
                         batch_size=batch_size,
                         learning_rate=learning_rate,
                         input_dims=input_dims,
                         output_dims=output_dims,
                         action_dims=action_dims,
                         verbose=verbose,
                         DEBUG=DEBUG,
                         model_dir=model_dir,
                         env_dims = env_dims,
                     ) 


    

    def evaluate_model(self, data_loader):
        total_error = jnp.zeros(self.output_dims)
        num_batches = 0

        for inputs, labels in data_loader:
            inputs = jnp.array(inputs)
            labels = jnp.array(labels)
            error = eval_step(self.state, inputs, labels)
            total_error += error 
            num_batches += 1

        avg_error = total_error / num_batches

        log_dict = {}
        for i in range(self.output_dims):
            log_dict[f"evaluating/validation_error{i}"] = avg_error[i]

        return log_dict


    def forward(self, x):
        # return self.model.apply(self.params, x)
        return self.state.apply_fn({'params': self.state.params}, x)

    def predict(self, x):
        ''' Assume x is numpy.array (batch, input_dims) - > (batch, output_dims)'''
        assert isinstance(x, np.ndarray)
        with torch.no_grad():
            # x = np.expand_dims(x,axis=0)
            x = (x - self._data_min) / (self._data_max - self._data_min + 1e-8)
            x = jnp.array(x)
            x = self.forward(x)
            x = x * (self._labels_max - self._labels_min + 1e-8) + self._labels_min
            # x = np.array(x)[0]
            x = np.array(x)
            assert x.shape[1] == self.output_dims
            return x

    @property
    def cbf_label_min(self):
        return self._labels_min

    @property
    def cbf_label_max(self):
        return self._labels_max

    def predict_affine(self, x):
        assert isinstance(x, np.ndarray)
        # __import__('pdb').set_trace()
        if len(x.shape) == 1:
            x = np.vstack((x,))
        x = (x - self._data_min) / (self._data_max - self._data_min + 1e-8)
        x = jnp.array(x)
        # print(f"shape input: {x.shape}")
        f_raw, g_raw = self.inference_model.apply({'params': {'model':self.state.params}}, x)
        # print(f"shape test, {f_raw.shape}, {g_raw.shape}")
        f_raw = np.array(f_raw).squeeze(axis=0)
        g_raw = np.array(g_raw).squeeze(axis=0)

        # x = np.array(x)
        # assert x.shape[1] == self.output_dims
        ret = {
            'f': f_raw,
            'g': g_raw,
            'y_min': self.cbf_label_min,
            'y_max': self.cbf_label_max,
            'x_min': self._data_min,
            'x_max': self._data_max,
        }
        return ret

    def save(self):
        # ckpt = {'params': self.params}
        ckpt = {'model': self.state, 
                'normalize_info':{
                                   'data_min':self._data_min,
                                   'data_max':self._data_max,
                                   'labels_min':self._labels_min,
                                   'labels_max':self._labels_max,
                               },
                }
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        # options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1, create=True)
        # checkpoint_manager = orbax.checkpoint.CheckpointManager(
        #     self.model_dir, orbax_checkpointer, options)
    
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(self.model_dir, ckpt, save_args=save_args)

    # checkpoint_manager.save(1, ckpt, save_kwargs={'save_args': save_args})

        # checkpoints.save_checkpoint(ckpt_dir=self.model_dir,
        #                     target=ckpt,
        #                     step=0,
        #                     overwrite=True,
        #                     keep=1)
        
    def load(self):
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        # raw_restored = orbax_checkpointer.restore(self.model_dir)
        print(f"[Info] Loading from {self.model_dir} ...")
        # raw_restored = checkpoints.restore_checkpoint(ckpt_dir=self.model_dir, target=None)
        raw_restored = orbax_checkpointer.restore(self.model_dir)
        # __import__('pdb').set_trace()
        # self.params = raw_restored['model']['params']
        self.state.params.update(raw_restored['model']['params'])
        # import pdb; pdb.set_trace()
        
        normalize_info = raw_restored['normalize_info']
        self.normalize_info(data_min = normalize_info['data_min'], 
                            data_max = normalize_info['data_max'], 
                            labels_min = normalize_info['labels_min'], 
                            labels_max = normalize_info['labels_max'])


        self.on_load_end()


    def normalize_info(self, data_min, data_max, labels_min, labels_max):
        self._data_min = data_min
        self._data_max =data_max
        self._labels_min = labels_min
        self._labels_max = labels_max



class EncDynamics(VanillaDynamics):
    def __init__(self, 
                 input_dims, 
                 output_dims, 
                 action_dims,
                 key1,
                 batch_size=1024,
                 learning_rate=1e-3, 
                 verbose = 0,
                 DEBUG = False,
                 model_dir = './', 
                 repr_dims = 4,
                 env_dims = 2,
                 ):
        super().__init__(input_dims, 
                         output_dims, 
                         action_dims, 
                         key1,
                         batch_size=batch_size,
                         learning_rate=learning_rate, 
                         verbose = verbose,
                         DEBUG=DEBUG,
                         model_dir=model_dir,
                         env_dims=env_dims)

        self.repr_dims = repr_dims

        self.model = AffineControlEncNN(input_dims=self.input_dims,
                                        output_dims=self.output_dims,
                                        action_dims=self.action_dims,
                                        env_dims=self.env_dims,
                                        repr_dims=self.repr_dims,
                                    )
        self.inference_model = AffineEncInference(self.model)
        
        self.key1, key2 = random.split(self.key1, 2)
        self.variable_inf = self.inference_model.init(key2, jnp.ones((batch_size, self.input_dims)))  

        self.key1, key2 = random.split(self.key1, 2)
        variable = self.model.init(key2, jnp.ones((batch_size, self.input_dims)))  

        optimizer = optax.adam(learning_rate)        # Create a State
        self.state = train_state.TrainState.create(
            apply_fn = self.model.apply,
            tx=optimizer,
            params=variable['params'],
        )

    def export_head(self, ) -> Dict:
        head_params = deepcopy(flax.core.unfreeze(self.state.params))
        head_params.pop('enc')

        return dict(
            model = self.model,
            params=head_params,
            normalize_info={
                'data_min':self._data_min,
                'data_max':self._data_max,
                'labels_min':self._labels_min,
                'labels_max':self._labels_max,
            },
        )
        

    def export_enc(self, ):
        enc_params = dict(enc=self.state.params['enc'])

        enc_info = dict(
            model = self.model,
            params=enc_params,
            normalize_info={
                'data_min':self._data_min[-self.action_dims-self.env_dims:-self.action_dims],
                'data_max':self._data_max[-self.action_dims-self.env_dims:-self.action_dims],
            },
        )

        self.key1, key2 = random.split(self.key1, 2)
        return BaseEnc1(input_dims=self.env_dims, 
                        enc_info=enc_info, 
                        key1=key2)


@ray.remote
def on_rollout_episode(params):
    env_generator, data_chunk, env_params_dim = params
    
    env = env_generator()
    X_l = []
    y_l = []
    
    step = 0
    while step < data_chunk:
        
        state_l = []
        action_l = []

        state = env.reset()
        state_l.append(state)
        
        while step < data_chunk:
            step += 1
            # print(step)
            action = env.action_space.sample()
            state, _, done, _ = env.step(action)
            action_l.append(action)
            state_l.append(state)
            if done:
                break

        state_l = np.array(state_l)
        action_l = np.array(action_l)

        for i in range(0, len(state_l)-1):
            dstate = state_l[i+1,:-env_params_dim] - state_l[i,:-env_params_dim]
            X = np.concatenate((state_l[i], action_l[i]), axis=0)
            X_l.append(X)
            y_l.append(dstate)

    return np.vstack(X_l), np.vstack(y_l)


class DynamicsLearning(BaseModelLearning):
    def __init__(self, model, env_generator, logger, DEBUG):

        
        super().__init__(model, env_generator, logger, DEBUG=DEBUG)


    def collect_rollouts(self,
                         dataset,
                         num_data,
                         policy,
                         data_chunk = 1000,
                         save_data_dir = None,
                         ):
        self.on_collect_rollouts_start()

        params = self.env_generator, data_chunk, self.model.env_dims 
        print(f"num data: {num_data}, data_chunk: {data_chunk}")
        futures = [on_rollout_episode.remote(params) for _ in range(num_data // data_chunk)]
        done = [] 
        # Function to track progress
        def track_progress(futures):
            while len(futures) > 0:
                done, futures = ray.wait(futures, num_returns=1, timeout=1.0)
                for _ in done:
                    yield

        # Use rich.progress.track to display progress
        for _ in track(track_progress(futures), 
                    description="Collecting data...", total=len(futures),disable=self.DEBUG):
            pass

        # Collect the results from workers
        results = ray.get(futures + done)

        # Append to dataset
        cnt_data = 0
        for X, Y in results:
            for x, y in zip(X, Y):
                cnt_data += 1
                # print(f"x shape: {x.shape}, y shape: {y.shape}")
                dataset.append(x, y)
        print("data size", len(dataset), "cnt", cnt_data)

        self.on_collect_rollouts_end(dataset, save_data_dir)


    def on_collect_rollouts_end(self,dataset, save_data_dir):
        create_missing_folder(save_data_dir)
        save_dataset(dataset, os.path.join(save_data_dir,'data.npz'))

    def on_train_start(self, dataset, ):
        # __import__('pdb').set_trace()
        data_min, data_max, labels_min, labels_max = normalize_dataset(dataset)
        self.model.normalize_info(data_min, data_max, labels_min, labels_max)
        print(colored(f"[INFO] On_train_start, data size: {len(dataset)}", 'blue'))

    # def on_collect_rollouts_loop(self,
    #                              env,
    #                              policy,
    #                              data_chunk,
    #                              window_length,
    #                          ):
    #
    #     X_l = []
    #     y_l = []
    #
    #     env.reset()
    #     traj_state = [env.robot_state4]
    #     traj_action = []
    #     for step in range(data_chunk):
    #         action = env.action_space.sample()
    #         _, _, done, _ = env.step(action)
    #         traj_state.append(env.robot_state4)
    #         traj_action.append(action)
    #         if done:
    #             break
    #
    #     for i in range(1, len(traj_state)):
    #         y = traj_state[i][:4] - traj_state[i-1][:4]
    #         X = np.concatenate((traj_state[i - 1], traj_action[i - 1]), axis=0)
    #         X_l.append(X)
    #         y_l.append(y)
    #
    #     return np.vstack(X_l), np.vstack(y_l)

    def on_evaluate_loop(self, 
                         env,
                         policy,
                         logger,
                         ):
        state = env.reset()
        X_l = []
        y_l = []
        for _ in itertools.count(0):
            self._total_testing_steps += 1 
            action = env.action_space.sample()

            # X = np.concatenate((env.robot_state4, action), axis=0)
            X = np.concatenate((state, action),axis=0)
            cur_loc = state[:-self.model.env_dims]

            state, _, done, _ = env.step(action)

            # delta_x = env.robot_state4[:4] - X[:4]
            delta_x = state[:-self.model.env_dims] - cur_loc
            X_l.append(X)
            y_l.append(delta_x)

            if done:
                break

        X_l = np.vstack(X_l)
        y_l = np.vstack(y_l)
        predict_delta = self.model.predict(X_l)
        # __import__('pdb').set_trace()
        error = np.abs((predict_delta - y_l) / (y_l + 1e-8) )
        delta_log = {'testing/total_test_step': self._total_testing_steps}
        for i in range(len(error)):
            for state_i in range(self.model.output_dims):
                delta_log[f"testing/delta{state_i}"] = predict_delta[i][state_i]
                delta_log[f"testing/deltaReal{state_i}"] = y_l[i][state_i]
                delta_log[f"testing/error{state_i}"] = error[i][state_i]
            logger.log(delta_log)

    def on_evaluate_loop2(self, 
                         env,
                         policy,
                         logger,
                         ):
        assert False

        '''Predict step by step.'''
        env.reset()
        X_l = []
        y_l = []
        for _ in itertools.count(0):
            self._total_testing_steps += 1 
            action = env.action_space.sample()

            X = np.concatenate((env.robot_state4, action), axis=0)
            # __import__('pdb').set_trace()
            X_l = np.expand_dims(X, axis=0)
            predict_delta = self.model.predict(X_l)

            _, _, done, _ = env.step(action)

            delta_x = env.robot_state4[:4] - X[:4]
            # X_l.append(X)
            # y_l.append(delta_x)
            print(f"pd: {predict_delta}, d: {delta_x}")

            if done:
                break

        # error = np.abs((predict_delta - y_l) / (y_l + 1e-8) )
        # delta_log = {'testing/total_test_step': self._total_testing_steps}
        # for i in range(len(error)):
        #     for state_i in range(self.model.output_dims):
        #         delta_log[f"testing/delta{state_i}"] = predict_delta[i][state_i]
        #         delta_log[f"testing/deltaReal{state_i}"] = y_l[i][state_i]
        #         delta_log[f"testing/error{state_i}"] = error[i][state_i]
        #     logger.log(delta_log)
