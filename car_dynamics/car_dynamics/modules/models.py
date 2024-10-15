import jax
import time
import numpy as np
import jax.numpy as jnp
import optax
from jax import random
# from torch.utils.data import random_split
from rich.progress import track

from modules.networks import PretrainedEnc
from .data_utils import NumpyLoader, FastLoader, random_split
from flax.training import train_state
from flax.core.frozen_dict import unfreeze


@jax.jit
def train_step(state, inputs, labels):
    def loss_fn(params):
        outputs = state.apply_fn({'params': params}, inputs)
        loss = jnp.mean((outputs - labels) ** 2)
        return loss, outputs

    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = gradient_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss 

@jax.jit
def eval_step(state, inputs, labels):
    outputs = state.apply_fn({'params': state.params}, inputs)
    return jnp.mean((outputs - labels) ** 2)

@jax.jit
def preconvert(inputs, labels):
    return jnp.array(inputs), jnp.array(labels)

class BaseModel1:
    ''' Assume integrated model '''

    def __init__(self,
                 input_dims,
                 output_dims,
                 encoder,
                 base_network,
                 inference_network,
                 key1,
                 batch_size=1024,
                 learning_rate=1e-3, 
                 verbose = 0,
                 DEBUG = False,
                 action_dims = 2,
                 model_dir = './',
                 env_dims = 0,
                 ):
        ''' Maintains model save/load/train/evaluate/inference '''
        self.encoder = encoder
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_dims = action_dims
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.key1 = key1
        self.env_dims = env_dims
        self.verbose =verbose
        self.DEBUG = DEBUG

        self._total_training_epoch = 0 

        self.model = base_network(input_dims=input_dims, 
                                  output_dims=output_dims,
                                  action_dims=action_dims,
                                  env_dims=env_dims,
                                )

        self.inference_model = inference_network(self.model)

        self.key1, key2 = random.split(self.key1, 2)
        self.variable_inf = self.inference_model.init(key2, jnp.ones((batch_size, self.input_dims)))  
        # print(self.variable_inf['params'])

        self.key1, key2 = random.split(self.key1, 2)
        variable = self.model.init(key2, jnp.ones((batch_size, self.input_dims)))  



        optimizer = optax.adam(learning_rate)        # Create a State
        self.state = train_state.TrainState.create(
            apply_fn = self.model.apply,
            tx=optimizer,
            params=variable['params'],
        )

        # self.params = unfreeze(params)
        # __import__('pdb').set_trace()

    
    @property
    def train_input_dims(self):
        return self.input_dims

    @property
    def train_output_dims(self):
        return self.output_dims


    def get_optimizer(self, lr):
        optimizer = optax.adam(lr)        # Create a State
        return optimizer

    def tune(self,
             logger,
             dataset,
             epoches,
             learning_rate,
             batch_size,
             tune_model_dir,
             data_split_rate=0.9):
        self.learning_rate = learning_rate
        self.model_dir = tune_model_dir
        self.batch_size = batch_size
        self.train(logger,dataset,epoches,data_split_rate)
        
    def train(self, 
              logger,
              dataset, 
              epoches, 
              data_split_rate = 0.8,
              ):
        '''Train and save model to `self.model_dir`
        '''

        # dataset.to_device()
        train_size = int(data_split_rate * dataset.length)  # Use 80% of the data for training
        eval_size = dataset.length - train_size
        train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

        if self.DEBUG:
            print("train size", train_size, eval_size)

        # print(f"[Info] train size: {train_size}, eval_size: {eval_size} ")
        # Create DataLoader objects for the training and evaluation sets
        # train_loader = NumpyLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # eval_loader = NumpyLoader(eval_dataset, batch_size=self.batch_size)
        # __import__('pdb').set_trace()
        
        # optimizer = optax.adam(self.learning_rate)        # Create a State
        tx = self.get_optimizer(self.learning_rate)
        self.state = train_state.TrainState.create(
            apply_fn = self.model.apply,
            tx=tx,
            # params=self.params['params']
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

            ## Train
            # t1 = time.time()

            train_log = self.train_model(train_loader, )

            ## Evaluate
            # t2 = time.time()
            eval_log = self.evaluate_model(eval_loader,)

            ## Others
            # t3 = time.time()
            log = {'train/total_epoch': self._total_training_epoch}
            log.update(train_log)
            log.update(eval_log)
            logger.log(log)
            # t4 = time.time()
            # print(f"prepare:{t1-t0}, d1:{t2-t1}, d2:{t3-t2}, d3:{t4-t3}, t4:{t4}")
        self.save()

    def train_model(self, data_loader):

        running_loss = []
        cnt_data = 0
        # t2 = time.time() 

        for inputs, labels in data_loader:
            # t1 = time.time()
            # loading_time = t1 - t2
            # print(f"tdata: {t1-t2}")
            # inputs, labels = preconvert(inputs, labels)           
            # t2 = time.time()

            self.state, loss = train_step(self.state, inputs, labels)
            running_loss.append(loss)
            # t2 = time.time()
            # print(f"[Train model] load:{loading_time}, d1:{t2-t1}, t2:{t2}")
            cnt_data += inputs.shape[0]

        running_loss = np.mean(running_loss)
        # print(f"[Info] meet {cnt_data} datapoints")
        return {'training/loss':running_loss}


    def evaluate_model(self, data_loader):
        running_loss = []

        for inputs, labels in data_loader:
            # inputs, labels = preconvert(inputs, labels)           
            loss = eval_step(self.state, inputs, labels)
            running_loss.append(loss)

        running_loss = np.mean(running_loss)

        return {'evaluating/loss':running_loss}

    def forward(self, x):
        ''' Forward for training / evaluate '''
        pass

    def predict(self, x):
        ''' Predict for testing '''
        pass


    def save(self):
        ''' Save the model, decides whether to include the encoder or not'''
        pass

    def on_load_end(self):
        pass

    def load(self):
        ''' Load the model, decides whether to include the encoder or not'''

        #### Implement Load Fn Here #######

        ####################################

        self.on_load_end()

    def normalize_info(self, ):
        pass


class BaseEnc1:
    def __init__(self,
                 input_dims,
                 enc_info,
                 key1,
                 ) -> None:

        self.input_dims = input_dims
        self.params = {'enc':enc_info['params']}
        self._data_min = enc_info['normalize_info']['data_min']
        self._data_max = enc_info['normalize_info']['data_max']
        self.key1 = key1
        
        self.model = PretrainedEnc(enc_info['model']) 

        self.key1, key2 = random.split(self.key1, 2)

        variable = self.model.init(key2, jnp.ones((1, self.input_dims)))  

        optimizer = optax.adam(1e-3)        # Create a State
        self.state = train_state.TrainState.create(
            apply_fn = self.model.apply,
            tx=optimizer,
            params=self.params,
        )



    def forward(self, x):
        # return self.model.apply(self.params, x)
        return self.state.apply_fn({'params': self.state.params}, x)

    def predict(self, x):
        ''' Assume x is numpy.array (batch, input_dims) - > (batch, output_dims)'''
        assert isinstance(x, np.ndarray)
        # x = np.expand_dims(x,axis=0)
        x = (x - self._data_min) / (self._data_max - self._data_min + 1e-8)
        x = jnp.array(x)
        x = self.forward(x)
        x = np.array(x)
        return x
