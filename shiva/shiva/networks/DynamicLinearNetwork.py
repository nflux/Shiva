import helpers.networks_handler as nh
import helpers.misc as misc
import torch

class DynamicLinearNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super(DynamicLinearNetwork, self).__init__()
<<<<<<< HEAD
        # print(config)
        #config = config['Critic']
        # print(config)
        # exit()
        self.net = nh.DynamicLinearSequential(
                            input_dim,
                            output_dim,
                            config['layers'],
                            nh.parse_functions(torch.nn,
                            config['activation_function']),
                            config['last_layer'],
                            misc.handle_package(torch.nn,
                            config['output_function'])
=======
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
>>>>>>> e90cd34a370540df209b41c2f3c0fbea8cb4d922
                            )
    def forward(self, x):
        return self.net(x)
