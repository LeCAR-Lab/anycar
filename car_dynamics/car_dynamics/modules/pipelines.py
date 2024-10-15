import numpy as np
import ray
from rich.progress import track
from .data_utils import DynDataset, load_dataset
from termcolor import colored
import os

class BaseModelLearning:
    def __init__(self,
                 model, # model to learn
                 env_generator, # gym environment Generator, env = env_generator()
                 logger, # wandb logger
                 verbose: int = 0, # 0:mute, 1:output information.
                 DEBUG: bool = False, # debug mode.
                 ) -> None:

        self.model = model
        self.env_generator = env_generator
        self.logger = logger

        self.verbose = verbose
        self.DEBUG = DEBUG


        self._total_testing_steps = 0 


    def learn(self,
              num_iterations: int = 1,
              rollout_policy = None,
              evaluate_policy = None,
              num_data: int = 1000_000,
              data_chunk = 500,
              train_epoches:int = 500,
              evaluate_episodes: int = 10,
              load_data = None,
              save_data_dir = None
              ):
        assert (save_data_dir is not None) or \
                (load_data is not None)
        
        for iter in range(num_iterations):

            if load_data is None:
                dataset = DynDataset(
                    input_dims=self.model.train_input_dims,
                    output_dims=self.model.train_output_dims,
                    max_length=num_data,
                )

                self.collect_rollouts(dataset, num_data, rollout_policy, data_chunk, save_data_dir=save_data_dir)
            else:
                dataset = load_dataset(os.path.join(load_data['file_name'],'data.npz'))
                print(colored(f"[Info] Loaded {dataset.length} data.", 'blue'))

            self.train(dataset, train_epoches)

            self.evaluate(evaluate_episodes, evaluate_policy)

            del dataset




    def collect_rollouts(self,
                         dataset,
                         num_data,
                         policy,
                         data_chunk = 1000,
                         ):
        self.on_collect_rollouts_start()

        
        ray.init(ignore_reinit_error=True, local_mode=True)
        # @ray.remote
        # def rollout_ep():
        #     env = self.env_generator()
        #     return self.on_collect_rollouts_loop(env, policy, data_chunk,window_length=1)
        params = self, policy, data_chunk, 1 
        futures = [BaseModelLearning.on_rollout_episode.remote(params) for _ in range(num_data // data_chunk)]
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

        # Collect the results
        results = ray.get(futures + done)

        # Append the results to the dataset
        print(f"[Info] Data Collected Shape: {len(results)}, {len(results[0])}")
        for x, y in results:
            dataset.append(x, y)

        self.on_collect_rollouts_end(dataset)


    def on_collect_rollouts_start(self,):
        pass

    def on_collect_rollouts_end(self,dataset):
        pass

    def on_collect_rollouts_loop(self,
                                 env,
                                 policy,
                                 data_chunk,
                                 window_length,
                             ):
        ''' Collect data and gather loop '''
        X_l = []
        y_l = []
        return np.vstack(X_l), np.vstack(y_l)


    def train(self,
              dataset,
              epoches,
              ):
        self.on_train_start(dataset)

        self.model.train(self.logger, dataset, epoches)


        self.on_train_end()

    def on_train_start(self, dataset):
        pass

    def on_train_end(self):
        pass

    def on_evaluate_loop(self, 
                         env,
                         policy,
                         logger):
        pass

    def evaluate(self, 
                 num_episodes, 
                 policy
                 ):

        env = self.env_generator()
        for i_ep in track(range(num_episodes), disable=self.DEBUG,
                          description='Evaluating Model ...'):
            self.on_evaluate_loop(env, policy, self.logger)

