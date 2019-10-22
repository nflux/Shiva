import helpers.networks_handler as nh
import torch.nn as nn

class DDPGActor(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super(DDPGActor, self).__init__()
        self.net = nh.DynamicLinearSequential(obs_dim, action_dim, 
                        nh.parse_layers(config['layers']), 
                        nh.parse_functions(config['activation_function']), 
                        config['last_layer'], 
                        nh.get_attrs(config['output_function']))
                        
    def forward(self, x):
        return self.net(x)