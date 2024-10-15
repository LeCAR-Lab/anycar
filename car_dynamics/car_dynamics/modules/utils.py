import os
import numpy as np
# import flax
from termcolor import colored
from typing import Any, Callable, Dict, Tuple
# from flax.core.scope import VariableDict
# from flax.traverse_util import _EmptyNode
# from flax.traverse_util import flatten_dict
# from flax.traverse_util import unflatten_dict

# Path = Tuple[str, ...]

# empty_node = _EmptyNode()

# https://github.com/google/flax/blob/a7673630d5cb88e1b8177427cdc713517419f78d/flax/traverse_util.py#L170
# def path_aware_map(
#     f: Callable[[Path, Any], Any], nested_dict: VariableDict
# ) -> VariableDict:
#   flat = flatten_dict(nested_dict, keep_empty_nodes=True)
#   return unflatten_dict(
#       {k: f(k, v) if v is not empty_node else v for k, v in flat.items()}
#   )

# def partition_params(params, frozen_path):
#     partitioned = path_aware_map(lambda path, v: 'frozen' if frozen_path in path else 'trainable',params)
#     return flax.core.freeze(partitioned)

def create_missing_folder(directory, is_file_name=False):
    if is_file_name:
        paths = directory.split('/')
        if not os.path.exists(os.path.join(paths[:-1])):
            os.makedirs(os.path.join(paths[:-1]))
            print(colored(f"[INFO] Create path:{os.path.join(paths[:-1])}", 'blue'))
    else:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(colored(f"[INFO] Create path:{directory}",'blue'))
            
def denormalize_angle_sequence(ang_seq):
    new_seq = [ang_seq[0]]
    counter = 0.
    for i in range(1, len(ang_seq)):
        diff = ang_seq[i] - ang_seq[i-1]
        if diff > np.pi:
            counter -= np.pi * 2.
        elif diff < -np.pi:
            counter += np.pi * 2.
        new_seq.append(ang_seq[i] + counter)
        
    return np.array(new_seq)