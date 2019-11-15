import helpers.networks_handler as nh
import helpers.misc as misc
import torch
from functools import partial
torch.manual_seed(5)

class DynamicLinearNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super(DynamicLinearNetwork, self).__init__()
        # print(config)
        self.config = config
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

class SoftMaxHeadDynamicLinearNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim, param_ix, config):
        '''
            Input
                @input_dim
                @output_dim
                @param_ix       Index from the output_dim where the parameters start
                @config
        '''
        super(SoftMaxHeadDynamicLinearNetwork, self).__init__()
        self.config = config
        self.param_ix = param_ix
        if self.config['gumbel']:
            self.softmax = partial(torch.nn.functional.gumbel_softmax, tau=1, hard=False, eps=1e-20, dim=-1)
        else:
            self.softmax = torch.nn.Softmax(dim=-1)
        
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
        o = self.net(x)
        # softmax = self.softmax(o[:, :, :self.param_ix])
        # params = o[:, :, self.param_ix:]
        return torch.cat([self.softmax(o[:, :, :self.param_ix]), o[:, :, self.param_ix:]], dim=2)
