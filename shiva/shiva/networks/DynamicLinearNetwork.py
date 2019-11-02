import helpers.networks_handler as nh
import helpers.misc as misc
import torch

class DynamicLinearNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super(DynamicLinearNetwork, self).__init__()
        # print(config)
        self.net = nh.DynamicLinearSequential(
                            input_dim,
                            output_dim,
                            # config['network']['layers'],
                            config['layers'],
                            # nh.parse_functions(torch.nn, config['network']['activation_function']),
                            nh.parse_functions(torch.nn, config['activation_function']),
                            # config['network']['last_layer'],
                            config['last_layer'],
                            # getattr(torch.nn, config['network']['output_function']) if config['network']['output_function'] is not None else None
                            getattr(torch.nn, config['output_function']) if config['output_function'] is not None else None
                        )
    def forward(self, x):
        return self.net(x)
