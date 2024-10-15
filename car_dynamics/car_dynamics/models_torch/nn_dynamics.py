import os
import time
import torch
import numpy as np
from car_foundation import CAR_FOUNDATION_MODEL_DIR, TorchTransformer, TorchTransformerDecoder, TorchGPT2
from termcolor import colored
def align_yaw(yaw_1, yaw_2):
    d_yaw = yaw_1 - yaw_2
    d_yaw_aligned = torch.atan2(torch.sin(d_yaw), torch.cos(d_yaw))
    return d_yaw_aligned + yaw_2

class DynamicsTorch:
    
    def __init__(self, params: dict):
        self.params = params
        latent_dim = 128
        num_heads = 4
        num_layers = 3
        dropout = 0.1
        self.model = TorchTransformerDecoder(6, 2, 6, latent_dim, num_heads, num_layers, dropout)
        # self.model = TorchGPT2(6, 2, 6, latent_dim, num_heads, num_layers, dropout)
        # self.model = TorchTransformer(8, 2, 6, 256, 8, 6, 0.1)
        # self.model = TorchTransformerDecoder(8, 2, 6, 256, 8, 6, 0.1)
        self.model.load_state_dict(torch.load(os.path.join(CAR_FOUNDATION_MODEL_DIR, "model.pth")))
        self.model.to('cuda:0')
        if params['is_dropout']:
            ...
        else:
            print(colored("[INFO] Turn off dropout", "green"))
            self.model.eval()
        
        # self.input_mean = torch.tensor([ 2.5802e-02, -3.0446e-04, -1.2580e-04,  6.4504e-01, -7.6116e-03, -3.1451e-03], device='cuda:0')
        # self.input_std = torch.tensor([0.1103, 0.0340, 0.0427, 2.7585, 0.8512, 1.0685], device='cuda:0')
        
        
        self.input_mean = torch.tensor([ 7.7609e-02, -3.7394e-04,  4.5594e-03,  1.9402e+00, -9.3484e-03, 1.1399e-01], device='cuda:0')
        self.input_std = torch.tensor([0.0368, 0.0065, 0.0274, 0.9201, 0.1615, 0.6848], device='cuda:0')
        
        
        
    def step(self, history, state, action):
        with torch.no_grad():
            st_nn_dyn = time.time()
            history = np.asarray(history)
            state = np.asarray(state)
            action = np.asarray(action)
            history = torch.from_numpy(history).cuda()
            state = torch.from_numpy(state).cuda()
            action = torch.from_numpy(action).cuda()
            # print(history.shape, action.shape)
            
            # batch shape: (batch_size, sequence_length + 1, 9)
            # convert pose state to delta
            original_yaw = history[:, :-1, 2].clone().detach()
            batch_tf = history.clone().detach()
            batch_tf[:, 1:, :3] = batch_tf[:, 1:, :3] - batch_tf[:, :-1, :3]
            batch_tf[:, 1:, 2] = align_yaw(batch_tf[:, 1:, 2], 0.0)
            # rotate dx, dy into body frame
            batch_tf_x = batch_tf[:, 1:, 0] * torch.cos(original_yaw) + batch_tf[:, 1:, 1] * torch.sin(original_yaw)
            batch_tf_y = -batch_tf[:, 1:, 0] * torch.sin(original_yaw) + batch_tf[:, 1:, 1] * torch.cos(original_yaw)
            batch_tf[:, 1:, 0] = batch_tf_x
            batch_tf[:, 1:, 1] = batch_tf_y
            batch_tf[:, :, :6] = ((batch_tf[:, :, :6] - self.input_mean) / self.input_std).detach()

            x = batch_tf[:, 1:, :]
            y_pred = self.model(x, action) * self.input_std + self.input_mean

            last_pose = history[:, -1, :3]
            for i in range(y_pred.shape[1]):
                # rotate dx, dy back to world frame
                y_pred_x = y_pred[:, i, 0] * torch.cos(last_pose[:, 2]) - y_pred[:, i, 1] * torch.sin(last_pose[:, 2])
                y_pred_y = y_pred[:, i, 0] * torch.sin(last_pose[:, 2]) + y_pred[:, i, 1] * torch.cos(last_pose[:, 2])
                y_pred[:, i, 0] = y_pred_x
                y_pred[:, i, 1] = y_pred_y
                # accumulate the poses
                y_pred[:, i, :3] += last_pose
                y_pred[:, i, 2] = align_yaw(y_pred[:, i, 2], 0.0)
                last_pose = y_pred[:, i, :3]
                
            print("NN Inference Time", time.time() - st_nn_dyn)
            return y_pred.cpu().detach().numpy()
        
        next_state = self.model(history, action)
        return next_state.cpu().detach().numpy()