import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import init, init_normc_
from model import Flatten
import gym

class ICM(nn.Module):
    def __init__(self, obs_shape, action_space, hidden_size=512, base_kwargs=None): 
        super(ICM, self).__init__()
        self.obs_shape = obs_shape # C x H x W
        self.action_space = action_space
        
        num_inputs = self.obs_shape[0]  
        num_outputs = self.action_space.n
      
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))
        
        # f(obs) = hidden_size
        self.phi = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )

        # f(phi_obs, action) = hidden_size
        num_inputs_forward_dynamic = hidden_size + num_outputs
        self.forward_dynamic = nn.Sequential(
            nn.Linear(num_inputs_forward_dynamic, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_size))

        # f(phi_obs, phi_obs_next) = num_outputs
        num_inputs_inverse_dynamic = hidden_size + hidden_size
        self.inverse_dynamic = nn.Sequential(
            nn.Linear(num_inputs_inverse_dynamic, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs))

        model_params = list(self.phi.parameters()) + list(self.forward_dynamic.parameters()) + list(self.inverse_dynamic.parameters())
        self.fwd_loss_func = nn.MSELoss(reduction='none')
        self.inv_loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model_params, lr=1e-3)

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
            
    def _forward_forward_dynamic(self, phi_obs, action):
        action_one_hot = torch.stack([torch.eye(self.num_outputs)[i].squeeze(0) for i in action]).cuda()
        x = torch.cat([phi_obs, action_one_hot], dim=1)
        y = self.forward_dynamic(x)
        return y

    def _forward_inverse_dynamic(self, phi_obs, phi_obs_next):
        x = torch.cat([phi_obs, phi_obs_next], dim=1)
        y = self.inverse_dynamic(x)       
        return y

    def forward(self, obs, action, obs_next):
        phi_obs = self.phi(obs / 255.0)
        phi_obs_next = self.phi(obs_next / 255.0)
        
        fwd_pred = self._forward_forward_dynamic(phi_obs, action)
        fwd_loss = self.fwd_loss_func(fwd_pred, phi_obs_next)
        
        return fwd_loss

    def train(self, obs, action, obs_next):
        phi_obs = self.phi(obs / 255.0)
        phi_obs_next = self.phi(obs_next / 255.0)
        
        fwd_pred = self._forward_forward_dynamic(phi_obs, action)
        inv_pred = self._forward_inverse_dynamic(phi_obs, phi_obs_next)
        fwd_loss = self.fwd_loss_func(fwd_pred, phi_obs_next.detach())
        inv_loss = self.inv_loss_func(inv_pred, action.squeeze(1))
        loss = fwd_loss.mean() + inv_loss
        
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()       

        return fwd_loss.sum(dim=1), inv_loss, loss
