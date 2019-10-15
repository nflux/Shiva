import torch
import torch.nn as nn
import numpy as np

def initialize_network(input_dim: int, output_dim: int, _params: dict):
    '''
        'Useful' function that returns a nn.Module network
    '''
    return DynamicLinearNetwork(
            input_dim = input_dim,
            output_dim = output_dim,
            layers = parse_layers(_params['layers']),
            activ_function = parse_functions(_params['activation_function']),
            output_function = get_attrs(_params['output_function'])
    )

#################################################################
###                                                           ###
###     Dynamic Linear Network Class                          ###
###                                                           ###
#################################################################

class DynamicLinearNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, layers: list, activ_function: list, output_function: object=None):
        super(DynamicLinearNetwork, self).__init__()
        self.net = DynamicLinearSequential(input_dim, output_dim, layers, activ_function, output_function)
    def forward(self, x):
        return self.net(x)

#################################################################
###                                                           ###
###     DDPG Actor & Critic Networks                          ###
###                                                           ###
#################################################################

class DDPGActor(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super(DDPGActor, self).__init__()
        self.net = DynamicLinearSequential(obs_dim, action_dim, parse_layers(config['layers']), parse_functions(config['activation_function']), config['last_layer'], get_attrs(config['output_function']))
    def forward(self, x):
        return self.net(x)

class DDPGCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, head_config, tail_config):
        super(DDPGCritic, self).__init__()
        self.net_head = DynamicLinearSequential(obs_dim, 400, parse_layers(head_config['layers']), parse_functions(head_config['activation_function']), head_config['last_layer'], get_attrs(head_config['output_function']))
        self.net_tail = DynamicLinearSequential(action_dim + 400, 1, parse_layers(tail_config['layers']), parse_functions(tail_config['activation_function']), tail_config['last_layer'],get_attrs(tail_config['output_function']) )
    def forward(self, x, a):
        obs = self.net_head(x)
        return self.net_tail(torch.cat([obs, a], dim=1))

#################################################################
###                                                           ###
###   Util functions                                          ###
###   by Ezequiel                                             ###
###                                                           ###
#################################################################

def parse_layers(layers_str):
    '''
        Input
            @layers_str     coming from the config as a string g.e. "20,10,20"
        Return
            List of int elements g.e. [20,10,20]
    '''
    return list(map(int, layers_str.split(',')))

def parse_functions(funcs_str, package=torch.nn):
    '''
        Input
            @func_str       coming from the config file as a string     g.e. "ReLU,ReLU,Tanh"
            @package        optional when using other package as TF
        Return
            List of function definitions        g.e. [nn.ReLU, nn.ReLU, nn.Tanh]
    '''
    return [get_attrs(f, package) for f in funcs_str.split(',')]

def get_attrs(func_str, package=torch.nn):
    '''
        This function is used by the parse_functions()
        
        Input
            @func_str       string name of a function     g.e. "ReLU"
        Return
            Function definition object (not instantiated)       g.e. nn.ReLU
    '''
    return getattr(package, func_str) if func_str != "" else None

def DynamicLinearSequential(input_dim, output_dim, layers: list, activ_function: list, last_layer:bool, output_function: object=None):
    '''
        Function that returns a nn.Sequential object determined by the inputs (essentially coming from config)
        obs_dim
        Input
            @input_dim
            @output_dim
            @layers             an iterable of layer sizes                                          g.e. [20,50,20]
            @activ_function     an iterable of function definitions to be used on each layer        g.e. [nn.ReLU, nn.ReLU, nn.Tanh]
            @output_function    a function definition that will transform the network output        g.e. nn.Tanh
    '''
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

    if last_layer:
        last_layer_dim = layers[len(layers)-1] if len(layers) > 1 else layers[0]
        net_layers.append( nn.Linear(last_layer_dim, output_dim) )
    
    
    if output_function is not None:
        net_layers.append(output_function())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return nn.Sequential(*net_layers).to(device)