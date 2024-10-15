import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import jax.numpy as jnp
import jax
from copy import deepcopy

class DynDataset(Dataset):
    def __init__(self, 
                 input_dims=4, 
                 output_dims=4, 
                 max_length=2000000):

        self.data = np.zeros((max_length, input_dims))
        self.labels = np.zeros((max_length, output_dims))
        self.length = 0

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def append(self, data_point, label):
        '''Assume input is numpy.array'''
        self.data[self.length] = data_point
        self.labels[self.length] = label
        self.length += 1

    # def to_device(self):
    #     self.data = jnp.array(self.data)
    #     self.labels = jnp.array(self.labels)


def save_dataset(dataset, filename):
    '''Helper function to save the dataset'''
    
    np.savez(filename, data=dataset.data, labels=dataset.labels, length=dataset.length)


def load_dataset(filename):
    '''Helper function to load the dataset'''

    data_file = np.load(filename) 
    data = np.array(data_file['data'])
    labels = np.array(data_file['labels'])
    length = int(data_file['length'])

    dataset = DynDataset()
    dataset.data = data
    dataset.labels = labels
    dataset.length = length
    return dataset

def convert_to_dataset(data, labels):
    dataset = DynDataset()
    length = data.shape[0]
    dataset.data = data
    dataset.labels = labels
    dataset.length = length
    return dataset

def normalize_dataset(dataset):
    data_min = np.min(dataset.data[:dataset.length], axis=0)
    data_max = np.max(dataset.data[:dataset.length], axis=0)
    labels_min = np.min(dataset.labels[:dataset.length], axis=0)
    labels_max = np.max(dataset.labels[:dataset.length], axis=0)

    dataset.data[:dataset.length] = (dataset.data[:dataset.length] - data_min) / \
                                                            (data_max - data_min + 1e-8)
    dataset.labels[:dataset.length] = (dataset.labels[:dataset.length] - labels_min) / \
                                                            (labels_max - labels_min + 1e-8)

    return data_min, data_max, labels_min, labels_max

def numpy_collate(batch):
    X_l = np.vstack([x for (x,y) in batch])
    y_l = np.vstack([y for (x,y) in batch])
  # if isinstance(batch[0], jnp.ndarray):
  #   return jnp.stack(batch)
  # elif isinstance(batch[0], (tuple,list)):
  #   transposed = zip(*batch)
  #   return [numpy_collate(samples) for samples in transposed]
  # else:
  #   return jnp.array(batch)
    # __import__('pdb').set_trace()
    # return jnp.array(batch)
    return X_l, y_l 

class NumpyLoader(DataLoader):
  def __init__(self, dataset, batch_size=1,
        # label_array = np.array(label, dtype=jnp.float32)
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)


def random_split(dataset, sizes):
    train_size, eval_size = sizes
    input_dims = dataset.data.shape[1]
    output_dims = dataset.labels.shape[1]
    trainset = DynDataset(input_dims,output_dims,train_size)
    evalset = DynDataset(input_dims, output_dims, eval_size)

    # __import__('pdb').set_trace()
    idx = np.random.permutation(dataset.length)
    trainset.data = dataset.data[idx[:train_size]]
    trainset.labels = dataset.labels[idx[:train_size]]
    trainset.length = train_size

    for i in range(train_size):
        if np.all(trainset.labels[i] == 0):
            print("error")

    evalset.data = dataset.data[idx[train_size:]]
    evalset.labels = dataset.labels[idx[train_size:]]
    evalset.length = eval_size

    assert evalset.data.shape[0] == eval_size

    return trainset, evalset


def FastLoader(dataset, batch_size, key1, shuffle=False):
    # __import__('pdb').set_trace()
    data, labels = jnp.array(dataset.data), jnp.array(dataset.labels)
    num_samples = dataset.length

    if shuffle:
        # Shuffle the indices
        key1, key2 = jax.random.split(key1, 2)
        perm_indices = jax.random.permutation(key2, num_samples)
        # print(perm_indices[:10])
        data = data[perm_indices]
        labels = labels[perm_indices]

    start_idx = num_samples % batch_size
    for i in range(start_idx, num_samples, batch_size):
        batch_data = data[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        yield batch_data, batch_labels

    # if start_idx != 0:
    #     batch_data = data[:batch_size]
    #     batch_labels = labels[:batch_size]
    #     yield batch_data, batch_labels
