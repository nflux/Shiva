import helpers.networks_handler as nh
import torch

class DDPGActor(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super(DDPGActor, self).__init__()
        self.net = nh.DynamicLinearSequential(obs_dim, action_dim, 
                        config['layers'], 
                        nh.parse_functions(nn, config['activation_function']), 
                        config['last_layer'], 
                        config['output_function']
                        )
                        
    def forward(self, x):
        return self.net(x)