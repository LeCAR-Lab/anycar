from typing import Union, Dict
import os
from .networks import AdaptationCNN, AdaptationNN, AdaptationDynamicsNN, AdaptationDynamicsNNInference
from .dynamics import VanillaDynamics
from .pipelines import BaseModelLearning
from jax import random
import jax.numpy as jnp
import optax
from termcolor import colored
import itertools
from flax.training import checkpoints
from flax.training import train_state
import flax
from flax import traverse_util
from .utils import partition_params, create_missing_folder
from .data_utils import normalize_dataset, save_dataset, random_split, NumpyLoader, FastLoader
import ray
import numpy as np
from rich.progress import track
import jax
from jax import lax

@jax.jit
def tune_all_step(state_adapt, state_head, state, history, action, labels, latent_min, latent_max):

    def loss_fn(params_adapt, params_head):
        latent = state_adapt.apply_fn({'params': params_adapt}, history)
        latent = latent * (latent_max - latent_min + 1e-8) + latent_min
        inputs_dyn = jnp.concatenate([state, latent, action], axis=-1) 
        outputs = state_head.apply_fn({'params': params_head}, inputs_dyn)
        loss = jnp.mean((outputs - labels) ** 2)
        return loss, (latent, outputs)

    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)

    params_adapt = state_adapt.params
    params_head = state_head.params

    (loss, _), grads_adapt, grads_head = gradient_fn(params_adapt, params_head)

    state_adapt = state_adapt.apply_gradients(grads=grads_adapt)
    state_head = state_head.apply_gradients(grads=grads_head)

    return state_adapt, state_head, loss


@jax.jit
def eval_step(state, inputs, labels):
    outputs = state.apply_fn({'params': state.params}, inputs)
    relative_error = jnp.abs((outputs - labels))
    return jnp.mean(relative_error, axis=0)
    
class AdaptationDynamics(VanillaDynamics):
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
                 env_dims = 10000,
                 env_params_dims=2,
                 repr_dims=4,
                 dyn_head_info:Union[Dict,None]=None,
                ):

        self.encoder = None
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_dims = action_dims
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.key1 = key1
        self.env_dims = env_dims
        self.env_params_dims = env_params_dims
        self.verbose =verbose
        self.DEBUG = DEBUG

        self._total_training_epoch = 0 

        self.repr_dims = repr_dims

        ## Load head
        self.dyn_model = dyn_head_info['model']
        self.dyn_params = dyn_head_info['params']
        self._dyn_labels_min = dyn_head_info['normalize_info']['labels_min']
        self._dyn_labels_max = dyn_head_info['normalize_info']['labels_max']
        self.dyn_data_min = dyn_head_info['normalize_info']['data_min']
        self.dyn_data_max = dyn_head_info['normalize_info']['data_max']

        adaptation_class = AdaptationCNN
        self.model = adaptation_class(input_dims=self.env_dims,
                                  output_dims=self.repr_dims)


        self.key1, key2 = random.split(self.key1, 2)
        variable = self.model.init(key2, jnp.ones((batch_size, self.env_dims)))  

        optimizer = optax.adam(learning_rate)        # Create a State
        self.state = train_state.TrainState.create(
            apply_fn = self.model.apply,
            tx=optimizer,
            params=variable['params'],
        )

    @property
    def train_input_dims(self):
        return self.env_dims

    @property
    def train_output_dims(self):
        return self.repr_dims

    def tune(self, mode: str, logger, dataset, epoches, learning_rate, batch_size,
            tune_model_dir, tune_head_dir, data_split_rate):
        '''Train and save model to `self.model_dir`
        '''
        self.model_dir = tune_model_dir
        # Load dynamics head to state
        optimizer = self.get_optimizer(self.learning_rate)
        self.head_state = train_state.TrainState.create(
            apply_fn = self.dyn_model.apply,
            tx = optimizer,
            params = self.dyn_params,
        )
        
        train_size = int(data_split_rate * dataset.length)  # Use 80% of the data for training
        eval_size = dataset.length - train_size
        train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

        if self.DEBUG:
            print("train size", train_size, eval_size)
        
        # optimizer = optax.adam(self.learning_rate)       # Create a State
        tx = self.get_optimizer(self.learning_rate)
        self.state = train_state.TrainState.create(
            apply_fn = self.model.apply,
            tx=tx,
            params=self.state.params,
        )

        if self.DEBUG:
            print("Training Start")

        for epoch in track(range(epoches), disable=self.DEBUG, description='Training Model ...'):

            ## Data Preparation
            # t0 = time.time()
            self.key1, key2 = random.split(self.key1, 2)
            train_loader = FastLoader(train_dataset, self.batch_size, key2, shuffle=True)
            self.key1, key2 = random.split(self.key1, 2)
            eval_loader = FastLoader(eval_dataset, self.batch_size, key2)

            self._total_training_epoch += 1

            train_log = self.tune_model(train_loader, mode)
            
            ## Evaluate
            # t2 = time.time()
            # eval_log = self.evaluate_model(eval_loader,)

            ## Others
            # t3 = time.time()
            log = {'train/total_epoch': self._total_training_epoch}
            log.update(train_log)
            log.update(eval_log)
            logger.log(log)
            # t4 = time.time()
            # print(f"prepare:{t1-t0}, d1:{t2-t1}, d2:{t3-t2}, d3:{t4-t3}, t4:{t4}")
        self.save()
        self.save_head(head_dir)

    def tune_model(self, data_loader, mode):

        running_loss = []
        cnt_data = 0

        for inputs, labels in data_loader:
            # print(inputs.shape)
            state, action = inputs[:,:-self.action_dims], inputs[:,-self.action_dims:]
            state, history = state[:,:-self.env_dims], state[:,-self.env_dims:]
            # print(history.shape)
            if mode == 'all':
                self.state, self.head_state, loss = tune_all_step(self.state, self.head_state, state, history, action, labels, 
                                                                  self._labels_min, self._labels_max)
            elif mode == 'adapt':
                self.state, self.head_state, loss = tune_enc_step(self.state, self.head_state, state, history, action, labels, 
                                                                  self._labels_min, self._labels_max)
            elif mode == 'head':
                self.state, self.head_state, loss = tune_head_step(self.state, self.head_state, state, history, action, labels, 
                                                                  self._labels_min, self._labels_max)
            else:
                raise Exception(f"Unkonwn mode: {mode}")
            running_loss.append(loss)
            cnt_data += inputs.shape[0]

        running_loss = np.mean(running_loss)
        # print(f"[Info] meet {cnt_data} datapoints")
        return {'tuning/loss': running_loss}
        
    def load_head(self, head_dir):
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        print(f"[Info] Loading from {head_dir} ...")

        raw_restored = orbax_checkpointer.restore(head_dir)
        self.dyn_data_min = raw_restored['normalize_info']['data_min']
        self.dyn_data_max = raw_restored['normalize_info']['data_max']
        self._dyn_labels_min = raw_restored['normalize_info']['labels_min']
        self._dyn_labels_max = raw_restored['normalize_info']['labels_max']
        
        self.dyn_model = raw_restored['model']
        self.dyn_params = raw_restored['model']['params']

    def save_head(self, head_dir):
        ckpt = {
            'model': self.head_state,
            'normalize_info': {
                    'data_min':self.dyn_data_min,
                    'data_max':self.dyn_data_max,
                    'labels_min':self._dyn_labels_min,
                    'labels_max':self._dyn_labels_max,
            }
        }
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(head_dir, ckpt, save_args=save_args)
        
    def evaluate_model(self, data_loader):
        total_error = jnp.zeros(self.repr_dims)
        num_batches = 0

        for inputs, labels in data_loader:
            inputs = jnp.array(inputs)
            labels = jnp.array(labels)
            error = eval_step(self.state, inputs, labels)
            total_error += error 
            num_batches += 1

        avg_error = total_error / num_batches

        return {
            f'evaluating/validation_mse_error{i}': avg_error[i] for i in range(self.repr_dims)
        }


    def deploy_dynamics(self,):

        self.adapt_dynamics = AdaptationDynamicsNN(
                            input_dims=self.input_dims,
                            output_dims=self.output_dims,
                            action_dims=self.action_dims,
                            env_dims=self.env_dims,
                            repr_dims=self.repr_dims,
                            model=self.dyn_model,
                            enc=self.model,
                            repr_min=self._labels_min,
                            repr_max=self._labels_max,
                    ) 

        self.inference_model = AdaptationDynamicsNNInference(model=self.dyn_model,
                                                             enc=self.model,
                                                             repr_min=self._labels_min,
                                                             repr_max=self._labels_max,
                                                            input_dims=self.input_dims,
                                                            output_dims=self.output_dims,
                                                            action_dims=self.action_dims,
                                                            env_dims=self.env_dims,
                                                            repr_dims=self.repr_dims,
                                                         )


        self.key1, key2 = random.split(self.key1, 2)
        variable_dyn = self.adapt_dynamics.init(key2, jnp.ones((1024, self.input_dims)))

        # __import__('pdb').set_trace()

        # __import__('pdb').set_trace()
        self.key1, key2 = random.split(self.key1, 2)
        self.variable_inf = self.inference_model.init(key2, jnp.ones((1024, self.input_dims)))  

        self._dyn_data_min = np.concatenate((
            self.dyn_data_min[:-self.env_params_dims-self.action_dims],
            self._data_min,
            self.dyn_data_min[-self.action_dims:]
        ),axis=0)

        self._dyn_data_max = np.concatenate((
            self.dyn_data_max[:-self.env_params_dims-self.action_dims],
            self._data_max,
            self.dyn_data_max[-self.action_dims:]
        ),axis=0)

    def on_load_end(self):
        self.deploy_dynamics()

    @property
    def cbf_label_min(self):
        ''' label_min for cbf '''
        return self._dyn_labels_min

    @property
    def cbf_label_max(self):
        ''' label_max for cbf '''
        return self._dyn_labels_max


    def predict_affine(self, x):
        assert isinstance(x, np.ndarray)
        # __import__('pdb').set_trace()
        if len(x.shape) == 1:
            x = np.vstack((x,))
        x = (x - self._dyn_data_min) / (self._dyn_data_max - self._dyn_data_min + 1e-8)
        x = jnp.array(x)
        # print(f"shape input: {x.shape}")
        f_raw, g_raw = self.inference_model.apply({'params': {'enc':self.state.params,
                                                  'model':self.dyn_params,
                                                  }}, x)
        # print(f"shape test, {f_raw.shape}, {g_raw.shape}")
        f_raw = np.array(f_raw).squeeze(axis=0)
        g_raw = np.array(g_raw).squeeze(axis=0)

        # x = np.array(x)
        # assert x.shape[1] == self.output_dims
        ret = {
            'f': f_raw,
            'g': g_raw,
            'x_min': self._dyn_data_min,
            'x_max': self._dyn_data_max,
            'y_min': self.cbf_label_min,
            'y_max': self.cbf_label_max,
        }
        return ret

    def predict_dynamics(self, x):
        assert isinstance(x, np.ndarray)
        # __import__('pdb').set_trace()
        if len(x.shape) == 1:
            x = np.vstack((x,))
        x = (x - self._dyn_data_min) / (self._dyn_data_max - self._dyn_data_min + 1e-8)
        x = jnp.array(x)
        # print(f"shape input: {x.shape}")
        x = self.adapt_dynamics.apply({'params': {'enc':self.state.params,
                                                  'model':self.dyn_params,
                                                  }}, x)
        x = x * (self._dyn_labels_max - self._dyn_labels_min + 1e-8) + self._dyn_labels_min
        
        x = np.array(x)
        assert x.shape[1] == self.output_dims
        return x


    # def get_optimizer(self, lr):
    #     optimizer = optax.adam(lr)
    #     return optimizer

    ## Depricated 
    # def get_optimizer(self, lr):
    #     partition_optimizers = {
    #         'trainable': optax.adam(lr), 
    #         'frozen': optax.set_to_zero()
    #     }
    #     param_partitions = partition_params(self.state.params, 'model')
    #     tx = optax.multi_transform(partition_optimizers,param_partitions)
    #
    #     return tx 



@ray.remote
def on_rollout_episode(params):
    env_generator, data_chunk, window_length, env_param_dim = params

    env = env_generator()
    X_l = []
    y_l = []

    step = 0
    while step < data_chunk:
        traj_state = []
        traj_action = []
        
        state = env.reset()
        traj_state.append(state)

        while step < data_chunk:
            step += 1
            action = env.action_space.sample()
            state, _, done, _ = env.step(action)
            traj_state.append(state)
            traj_action.append(action)
            if done:
                break

        traj_state = np.array(traj_state)
        traj_action = np.array(traj_action)

        for i in range(window_length, len(traj_state)):
            y = traj_state[i-1][-env_param_dim:]
            # __import__('pdb').set_trace()
            X = np.concatenate((traj_state[i-window_length:i][:,:-env_param_dim], 
                                traj_action[i-window_length:i]), axis=1)
            # __import__('pdb').set_trace()
            X = np.concatenate(X, axis=0)
            X = X[:-traj_action.shape[1]]
            # X = np.concatenate((traj_state[i-1], X, traj_action[i-1]), axis=0)
            X_l.append(X)
            y_l.append(y)

    return np.vstack(X_l), np.vstack(y_l)

class AdaptationLearning(BaseModelLearning):
    def __init__(self, 
                 model, 
                 env_generator, 
                 logger, 
                 DEBUG,
                 window_length,
                 target_enc,
             ) -> None:

        
        super().__init__(model=model, env_generator=env_generator, logger=logger, DEBUG=DEBUG)

        self.window_length = window_length
        self.target_enc = target_enc

    def collect_rollouts(self,
                         dataset,
                         num_data,
                         policy,
                         data_chunk = 1000,
                         save_data_dir = None,
                         ):
        self.on_collect_rollouts_start()

        params = self.env_generator, data_chunk, self.window_length, self.model.env_params_dims
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
        X_l = []
        y_l = []
        for X, Y in results:
            for x, y in zip(X, Y):
                X_l.append(x)
                y_l.append(y)

        ##Inference
        y_l = np.array(y_l)
        # __import__('pdb').set_trace()
        y_l = self.target_enc.predict(y_l)

        # __import__('pdb').set_trace()
        for x, y in zip(X_l, y_l):
            dataset.append(x, y)

        self.on_collect_rollouts_end(dataset, save_data_dir)

    def on_collect_rollouts_end(self,dataset, save_data_dir):
        create_missing_folder(save_data_dir)
        save_dataset(dataset, os.path.join(save_data_dir,'data.npz'))

    def on_train_start(self, dataset):
        # save_dataset(dataset, './adapt_data.npz')
        data_min, data_max, labels_min, labels_max = normalize_dataset(dataset)
        self.model.normalize_info(data_min, data_max, labels_min, labels_max)
        print(colored(f"[INFO] On_train_start, data size: {len(dataset)}", 'blue'))

    def on_evaluate_loop(self, 
                         env,
                         policy,
                         logger,
                         ) -> None:

        X_l = []
        y_l = []
        traj_state = []
        traj_action = []
        
        state = env.reset()
        # traj_state = [env.robot_state4[:-ENV_PARAM_DIM]]
        traj_state.append(state[:-self.model.env_params_dims])
        for _ in itertools.count(0):
            self._total_testing_steps += 1 
            action = env.action_space.sample()
            state, _, done, _ = env.step(action)
            # traj_state.append(env.robot_state4[:-ENV_PARAM_DIM])
            traj_state.append(state[:-self.model.env_params_dims])
            traj_action.append(action)
            if done:
                break

        traj_action = np.array(traj_action)

        for i in range(self.window_length, len(traj_state)):
            y = traj_state[i][:self.model.output_dims] - traj_state[i-1][:self.model.output_dims]
            X = np.concatenate((traj_state[i-self.window_length:i], 
                                traj_action[i-self.window_length:i]), axis=1)
            X = np.concatenate(X, axis=0)
            X = X[:-traj_action.shape[1]]
            X = np.concatenate((traj_state[i-1], X, traj_action[i-1]), axis=0)
            X_l.append(X)
            y_l.append(y)

        X_l = np.vstack(X_l)
        y_l = np.vstack(y_l)

        self.model.deploy_dynamics()
        predict_delta = self.model.predict_dynamics(X_l)
        error = np.abs((predict_delta - y_l) / (y_l + 1e-8) )
        delta_log = {'testing/total_test_step': self._total_testing_steps}
        for i in range(len(error)):
            for state_i in range(self.model.output_dims):
                delta_log[f"testing/delta{state_i}"] = predict_delta[i][state_i]
                delta_log[f"testing/deltaReal{state_i}"] = y_l[i][state_i]
                delta_log[f"testing/error{state_i}"] = error[i][state_i]
            logger.log(delta_log)

        summary = {}
        # import pdb; pdb.set_trace()
        # print(error[:, 0].shape, np.mean(error[:, 0]))
        for i in range(self.model.output_dims):
            summary[f'summary/error{i}_mean']= np.mean(error[:, i])
            summary[f'summary/error{i}_max']= np.max(error[:, i])
            summary[f'summary/error{i}_min']= np.min(error[:, i])

        logger.log(summary)


    def on_evaluate_loop_outside(self, 
                         env,
                         policy,
                         logger,
                         ) -> None:
        '''Call from outside'''
        assert False

        X_l = []
        y_l = []
        env.reset()
        traj_state = [env.robot_state4[:-ENV_PARAM_DIM]]
        traj_action = []
        for _ in itertools.count(0):
            self._total_testing_steps += 1 
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
            traj_state.append(env.robot_state4[:-ENV_PARAM_DIM])
            traj_action.append(action)
            if done:
                break

        traj_action = np.array(traj_action)

        for i in range(self.window_length, len(traj_state)):
            y = traj_state[i][:4] - traj_state[i-1][:4]
            X = np.concatenate((traj_state[i-self.window_length:i], 
                                traj_action[i-self.window_length:i]), axis=1)
            X = np.concatenate(X, axis=0)
            X = X[:-traj_action.shape[1]]
            X = np.concatenate((traj_state[i-1], X, traj_action[i-1]), axis=0)
            X_l.append(X)
            y_l.append(y)

        X_l = np.vstack(X_l)
        y_l = np.vstack(y_l)

        self.model.deploy_dynamics()
        predict_delta = self.model.predict_dynamics(X_l)
        error = np.abs((predict_delta - y_l) / (y_l + 1e-8) )
        delta_log = {'testing/total_test_step': self._total_testing_steps}
        for i in range(len(error)):
            for state_i in range(self.model.output_dims):
                delta_log[f"testing/delta{state_i}"] = predict_delta[i][state_i]
                delta_log[f"testing/deltaReal{state_i}"] = y_l[i][state_i]
                delta_log[f"testing/error{state_i}"] = error[i][state_i]
            logger.log(delta_log)

        summary = {}
        for i in range(self.model.output_dims):
            summary[f'summary/error{i}_mean']= np.mean(error[:, i])
            summary[f'summary/error{i}_max']= np.max(error[:, i])
            summary[f'summary/error{i}_min']= np.min(error[:, i])

        logger.log(summary)
