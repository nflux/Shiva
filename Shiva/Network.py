import torch
import torch.nn as nn
import numpy as np

def initialize_network(input_dim: int, output_dim: int, _params: dict):
    return DynamicLinearNetwork(
            input_dim = input_dim,
            output_dim = output_dim,
            layers = list(map(int, _params['layers'].split(','))),
            activ_function = [getattr(torch.nn, f) for f in _params['activation_function'].split(',')],
            output_function = getattr(torch.nn, _params['output_function']) if _params['output_function'] != "" else None
    )

#################################################################
###                                                           ###
###     Dynamic Linear Network Class                          ###
###     extends torch.nn.Module                               ###
###                                                           ###
#################################################################

class DynamicLinearNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, layers: list, activ_function: list, output_function: object=None):
        super(DynamicLinearNetwork, self).__init__()
        assert hasattr(layers, '__iter__'), '@layers input must be iterable - try a list or set'
        assert len(layers) > 0, '@layers input is empty'
        assert len(activ_function) == 1 or (len(activ_function) > 1 and len(activ_function) == len(layers)), '@layers and @activ_function must be same size if more than one @activ_function is given'
        
        net_layers = []
        net_layers.append(nn.Linear(input_dim, layers[0]))
        net_layers.append(activ_function[0]())
        in_layer_size = layers[0]
        
        if len(activ_function) > 1:
            for i, (out_layer_size, af) in enumerate(zip(layers, activ_function)):
                if i == 0: continue
                net_layers.append(nn.Linear(in_layer_size, out_layer_size))
                net_layers.append(af())
                in_layer_size = out_layer_size
        else:
            af = activ_function[0]
            for i, out_layer_size in enumerate(layers):
                if i == 0: continue
                net_layers.append(nn.Linear(in_layer_size, out_layer_size))
                net_layers.append(af())
                in_layer_size = out_layer_size
                
        last_layer_dim = layers[len(layers)-1] if len(layers) > 1 else layers[0]
        net_layers.append( nn.Linear(last_layer_dim, output_dim) )
        
        if output_function is not None:
            net_layers.append(output_function())

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(*net_layers).to(device)
    
    def forward(self, x):
        return self.net(x)
