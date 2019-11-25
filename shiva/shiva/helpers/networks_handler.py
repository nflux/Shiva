import helpers.misc as misc
import torch
import torch.nn as nn

# def parse_layers(layers_str):
#     '''
#         Input
#             @layers_str     coming from the config as a string g.e. "20,10,20"
#         Return
#             List of int elements g.e. [20,10,20]
#     '''
#     return list(map(int, layers_str))

def parse_functions(package, funcs_str):
    '''
        Input
            @func_str       coming from the config file as a string     g.e. "ReLU,ReLU,Tanh"
            @package        optional when using other package as TF
        Return
            List of function definitions        g.e. [nn.ReLU, nn.ReLU, nn.Tanh]
    '''
    return [misc.handle_package(package, f) for f in funcs_str]

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

    print("Networks Handler: ", input_dim, layers[0])

    print(activ_function)

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