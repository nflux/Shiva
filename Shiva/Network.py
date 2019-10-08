import torch
import torch.nn as nn
import numpy as np

def initialize_network(input_dim: int, output_dim: int, _params: dict):
    return DynamicLinearNetwork(
            input_dim = input_dim,
            output_dim = output_dim,
            layers = list(map(int, _params['layers'].split(','))),
            activ_function = getattr(torch.nn, _params['activation_function'])
    )

#################################################################
###                                                           ###
###     Dynamic Network Class                                 ###
###     extends torch.nn.Module                               ###
###                                                           ###
#################################################################
class DynamicLinearNetwork(nn.Module):
    def __init__(self, 
                input_dim: int, 
                output_dim: int, 
                layers: list, 
                activ_function: object=nn.ReLU
        ):
        '''
            Inputs
                input_dim
                output_dim
                layers:             iterable containing layer sizes
                activ_function      for now is the activation function used at every layer
        '''
        super(DynamicLinearNetwork, self).__init__()
        print(layers[0])
        assert hasattr(layers, '__iter__'), 'Layers input must be iterable - try a list or set'
        assert len(layers) > 0, 'Layers input is empty'
        net_layers = []
        net_layers.append(nn.Linear(input_dim, layers[0]))
        net_layers.append(activ_function())
        for i, l in enumerate(layers):
            if i == 0: continue
            net_layers.append(nn.Linear(layers[i-1], layers[i]))
            net_layers.append(activ_function())
        last_layer_dim = layers[len(layers)-1] if len(layers) > 1 else layers[0]
        net_layers.append( nn.Linear(last_layer_dim, output_dim) ) 

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(*net_layers).to(device)
    
    def forward(self, x):
        return self.net(x)

# class Network(nn.Module):
#     def __init__(self, input_shape, output_shape):
#         super(Network, self).__init__()
#         self.input_shape = input_shape
#         self.output_shape = output_shape
    
#     def forward(self):
#         pass


#################################################################
###                                                           ###
###     DQ Network Class                                      ###
###                                                           ###
#################################################################
# class DQNet(Network):
#     def __init__(self, input_size, HIDDEN_SIZE_1, HIDDEN_SIZE_2, output_size):
#         super(DQNet, self).__init__(input_size, output_size)
#         self.net = nn.Sequential(
#             nn.Linear(input_size, HIDDEN_SIZE_1),
#             nn.ReLU(),
#             nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
#             nn.ReLU(),
#             nn.Linear(HIDDEN_SIZE_2, output_size),
#         )
        
#     def forward(self, x):
#         return self.net(x)
   
