import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any

class Model(nn.Module):
    def __init__(self, ):
        super(Model, self).__init__()
        self.spec = {
            'normalization': None,
        }
    
    def forward(self, x):
        raise NotImplementedError
           
    def save(self, path):
        # Combine the model state dict and the spec dict
        combined_dict = {
            'state_dict': self.state_dict(),
            'spec': self.spec
        }
        # Save the combined dictionary
        torch.save(combined_dict, path)

    def load(self, path, map_location=None):
        if map_location is None:
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
    
        # Load the combined dictionary
        combined_dict = torch.load(path, map_location=map_location)
        
        # Set the model state dict
        self.load_state_dict(combined_dict['state_dict'])
        
        # Update the model's spec dictionary
        self.spec = combined_dict['spec']
        
    def update_spec(self, spec: Dict):
        self.spec.update(spec)
   
    def predict(self, x):
        with torch.no_grad():
            if self.spec['normalization'] is None:
                return self.forward(x)
            elif self.spec['normalization'] == 'minmax':
                assert self.spec['input_min'] is not None
                assert self.spec['input_max'] is not None
                assert self.spec['output_min'] is not None
                assert self.spec['output_max'] is not None
                x = (x - self.spec['input_min']) / (self.spec['input_max'] - self.spec['input_min'] + 1e-8)
                x = self.forward(x)
                x = x * (self.spec['output_max'] - self.spec['output_min'] + 1e-8) + self.spec['output_min']
                return x
            elif self.spec['normalization'] == 'z-score':
                assert self.spec['input_mean'] is not None
                assert self.spec['input_std'] is not None
                assert self.spec['output_mean'] is not None
                assert self.spec['output_std'] is not None
                x = (x - self.spec['input_mean']) / (self.spec['input_std'] + 1e-8)
                x = self.forward(x)
                x = x * (self.spec['output_std'] + 1e-8) + self.spec['output_mean']
                return x
                
            else:
                raise NotImplementedError

class MLP(Model):
    def __init__(self, input_size, hidden_size, output_size, 
                 last_layer_activation='none', gru_hidden_size=0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.last_layer_activation = last_layer_activation

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.last_layer_activation == 'tanh':
            x = F.tanh(x)
        return x

    
                
      
        
class GRUMLP(Model):
    def __init__(self, input_size, hidden_size, output_size, gru_hidden_size,
                 last_layer_activation='none'):
        super(GRUMLP, self).__init__()
        self.gru = nn.GRU(input_size, gru_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(gru_hidden_size, hidden_size) # Adjusted for GRU output
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.last_layer_activation = last_layer_activation
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        # import pdb; pdb.set_trace()
        # gru_out = gru_out[:, -1, :]
        x = F.relu(self.fc1(gru_out))
        x = self.fc2(x)
        if self.last_layer_activation == 'tanh':
            x = F.tanh(x)
        return x