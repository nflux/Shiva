import torch
from functools import partial

from shiva.helpers import networks_handler as nh

class DynamicLinearNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super(DynamicLinearNetwork, self).__init__()
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
        self.gumbel = partial(torch.nn.functional.gumbel_softmax, tau=1, hard=True, dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)

        assert type(output_dim) == tuple, "Expecting output dimension to be type tuple. Do (5,) for single discrete action and (5,2,..) for multidiscrete"

        if len(output_dim) == 1:
            # Single branch of discrete actions
            self.shared_net = nh.DynamicLinearSequential(
                input_dim,
                output_dim=output_dim[0],
                layers=config['layers'],
                activ_function=nh.parse_functions(torch.nn, config['activation_function']),
                last_layer=True,  # config['last_layer'],
                output_function=getattr(torch.nn, config['output_function']) if config['output_function'] is not None else None
            )
            self.branch_net = []
            self.forward = self._forward_single_branch
        else:
            # Multiple branches of discrete actions
            self.shared_net = nh.DynamicLinearSequential(
                input_dim,
                output_dim=None,
                layers=config['layers'][:-1],
                activ_function=nh.parse_functions(torch.nn, config['activation_function'][:-1]),
                last_layer=False,  # config['last_layer'],
                output_function=getattr(torch.nn, config['output_function']) if config['output_function'] is not None else None
            )
            self.branch_net = [
                nh.DynamicLinearSequential(
                    config['layers'][-2],
                    output_dim=ac_dim,
                    layers=config['layers'][-1][ac_ix],
                    activ_function=nh.parse_functions(torch.nn, config['activation_function'][-1][ac_ix]),
                    last_layer=True,  # config['last_layer'],
                    output_function=getattr(torch.nn, config['output_function']) if config['output_function'] is not None else None
                )
                for ac_ix, ac_dim in enumerate(output_dim)
            ]
            self.forward = self._forward_multi_branch

    def _forward_multi_branch(self, x, gumbel=False):
        _shared_output = self.shared_net(x)
        if gumbel:
            return torch.cat([self.gumbel(_branch_nn(_shared_output)) for _branch_nn in self.branch_net], dim=-1)
        else:
            return torch.cat([self.softmax(_branch_nn(_shared_output)) for _branch_nn in self.branch_net], dim=-1)

    def _forward_single_branch(self, x, gumbel=False, onehot=False):
        a = self.shared_net(x)
        if gumbel:
            # print("Network Output (before gumbel):", x, a)
            if len(a.shape) == 3:
                return torch.cat([self.gumbel(a[:, :, :self.param_ix]), a[:, :, self.param_ix:]], dim=2)
            elif len(a.shape) == 2:
                return torch.cat([self.gumbel(a[:, :self.param_ix]), a[:, self.param_ix:]], dim=1)
            else:
                return torch.cat([self.gumbel(a[:self.param_ix]), a[self.param_ix:]], dim=0)
        elif onehot:
            # print("Network Output (before gumbel):", x, a)
            if len(a.shape) == 3:
                return torch.cat([self.gumbel(a[:, :, :self.param_ix]), a[:, :, self.param_ix:]], dim=2)
            elif len(a.shape) == 2:
                return torch.cat([self.gumbel(a[:, :self.param_ix]), a[:, self.param_ix:]], dim=1)
            else:
                return torch.cat([self.gumbel(a[:self.param_ix]), a[self.param_ix:]], dim=0)
        else:
            # print("Network Output (before softmax):", x, a)
            if len(a.shape) == 3:
                return torch.cat([self.softmax(a[:, :, :self.param_ix]), a[:, :, self.param_ix:]], dim=2)
            elif len(a.shape) == 2:
                return torch.cat([self.softmax(a[:, :self.param_ix]), a[:, self.param_ix:]], dim=1)
            else:
                return torch.cat([self.softmax(a[:self.param_ix]), a[self.param_ix:]], dim=0)
