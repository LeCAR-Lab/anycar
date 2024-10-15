'''Safety Method Baselines
- Mix
- Fix
'''

from .dynamics import DynamicsLearning
import ray
from rich.progress import track
import numpy as np
import itertools

ENV_PARAM_DIM = 2

@ray.remote
def on_rollout_episode(params):
    env_generator, data_chunk = params

    env = env_generator()
    X_l = []
    y_l = []

    env.reset()
    traj_state = [env.robot_state4[:-ENV_PARAM_DIM]]
    traj_action = []
    for step in range(data_chunk):
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        traj_state.append(env.robot_state4[:-ENV_PARAM_DIM])
        traj_action.append(action)
        if done:
            break

    for i in range(1, len(traj_state)):
        y = traj_state[i][:4] - traj_state[i-1][:4]
        X = np.concatenate((traj_state[i - 1], traj_action[i - 1]), axis=0)
        X_l.append(X)
        y_l.append(y)

    return np.vstack(X_l), np.vstack(y_l)

class MFixLearning(DynamicsLearning):
    def __init__(self,
                 model, 
                 env_generator, 
                 logger, 
                 DEBUG,
                 ) -> None:
        
        super().__init__(model, env_generator, logger, DEBUG)

    
    def collect_rollouts(self,
                         dataset,
                         num_data,
                         policy,
                         data_chunk = 1000,
                         ):
        self.on_collect_rollouts_start()

        params = self.env_generator, data_chunk 
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
        for X, Y in results:
            for x, y in zip(X, Y):
                dataset.append(x, y)

        self.on_collect_rollouts_end()

    def on_evaluate_loop(self, 
                         env,
                         policy,
                         logger,
                         ):
        env.reset()
        X_l = []
        y_l = []
        for _ in itertools.count(0):
            self._total_testing_steps += 1 
            action = env.action_space.sample()

            X = np.concatenate((env.robot_state4[:-ENV_PARAM_DIM], action), axis=0)

            _, _, done, _ = env.step(action)

            delta_x = env.robot_state4[:4] - X[:4]
            X_l.append(X)
            y_l.append(delta_x)

            if done:
                break

        X_l = np.vstack(X_l)
        y_l = np.vstack(y_l)
        predict_delta = self.model.predict(X_l)
        error = np.abs((predict_delta - y_l) / (y_l + 1e-8) )
        delta_log = {'testing/total_test_step': self._total_testing_steps}
        for i in range(len(error)):
            for state_i in range(self.model.output_dims):
                delta_log[f"testing/delta{state_i}"] = predict_delta[i][state_i]
                delta_log[f"testing/deltaReal{state_i}"] = y_l[i][state_i]
                delta_log[f"testing/error{state_i}"] = error[i][state_i]
            logger.log(delta_log)
