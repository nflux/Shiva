import helpers.networks_handler as nh
import helpers.misc as misc
import torch

class DynamicLinearNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super(DynamicLinearNetwork, self).__init__()
        self.net = nh.DynamicLinearSequential(input_dim, output_dim, 
                            nh.parse_layers(config['layers']), nh.parse_functions(torch.nn, config['activation_function']), 
                            misc.handle_package(torch.nn, config['output_function']))
    def forward(self, x):
        return self.net(x)