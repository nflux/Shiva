import helpers.networks_handler as nh
import torch.nn as nn

class DDPGCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, head_config, tail_config):
        super(DDPGCritic, self).__init__()
        self.net_head = nh.DynamicLinearSequential(obs_dim, 
                                                400, 
                                                nh.parse_layers(head_config['layers']), 
                                                nh.parse_functions(head_config['activation_function']), 
                                                head_config['last_layer'], 
                                                nh.get_attrs(head_config['output_function']))
                                                
        self.net_tail = nh.DynamicLinearSequential(action_dim + 400, 
                                                1, 
                                                nh.parse_layers(tail_config['layers']), 
                                                nh.parse_functions(tail_config['activation_function']), 
                                                tail_config['last_layer'],
                                                nh.get_attrs(tail_config['output_function']) )
    
    def forward(self, x, a):
        obs = self.net_head(x)
        return self.net_tail(torch.cat([obs, a], dim=1))