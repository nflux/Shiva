import shiva.helpers.misc as misc
import torch
import torch.nn as nn


def parse_functions(package, funcs_str):
    """
    Is this being used?????

    Args:
        package: package name
        funcs_str: coming from the config file as a string     g.e. "ReLU,ReLU,Tanh"

    Returns:
        List of functions definitions

    Example:
        >>> parse_functions('torch.nn', ["ReLU", "ReLU", "Tanh"])
        [nn.ReLU, nn.ReLU, nn.ReLU]
    """
    return [misc.handle_package(package, f) for f in funcs_str]


def DynamicLinearSequential(input_dim, output_dim, layers: list, activ_function: list, last_layer:bool, output_function: object=None):
    """
    Creates a new dynamic nn.Sequential network. The size of `layers` must be equal to the size of `activ_function`.

    Args:
        input_dim (int): the input dimension
        output_dim (int): the output dimension
        layers (List[int]): list of the size of the hidden layers
        activ_function (List[object]): list of activation function objects for each of the layers
        last_layer (bool): boolean stating if we include the last layer to the network
        output_function (object): Optional. Output function object.

    Returns:
        torch.nn.Sequential
    """
    assert hasattr(layers, '__iter__'), f"@layers input must be iterable - try a list or set, got {layers}"
    assert len(layers) > 0, f"@layers input is empty, got {layers}"
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


    for i, layer in enumerate(net_layers):
        if 'linear' in str(type(layer)):
            layer.weight.data.normal_(0, 0.01) 
        # if i % 2 == 0:
        #     net_layers[i] = 

    # input()

    if output_function is not None:
        try:
            net_layers.append(output_function(dim=-1))
        except TypeError:
            net_layers.append(output_function())

    return nn.Sequential(*net_layers)
