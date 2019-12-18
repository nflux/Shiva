# import helpers.networks_handler as nh
import torch.nn as nn
import torch
from networks.DynamicLinearNetwork import  SoftMaxHeadDynamicLinearNetwork

# SoftMaxHeadDynamicLinearNetwork

class DDPGActor(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super(DDPGActor, self).__init__()
        self.net = SoftMaxHeadDynamicLinearNetwork(obs_dim, action_dim, action_dim,
                            config
                        )
                        
    def forward(self, x):
        return self.net(x)