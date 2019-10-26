import helpers.networks_handler as nh
import helpers.misc as misc
import torch

class DynamicLinearNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super(DynamicLinearNetwork, self).__init__()
        print(config)
        self.net = nh.DynamicLinearSequential(
                            input_dim, 
                            output_dim, 
                            config['network']['layers'], 
                            nh.parse_functions(torch.nn, 
                            config['network']['activation_function']),
                            config['network']['last_layer'], 
                            misc.handle_package(torch.nn, 
                            config['network']['output_function'])
                            )
    def forward(self, x):
        return self.net(x)
