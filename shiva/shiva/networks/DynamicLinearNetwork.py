import helpers.networks_handler as nh
import torch.nn as nn

class DynamicLinearNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, layers: list, activ_function: list, output_function: object=None):
        super(DynamicLinearNetwork, self).__init__()
        self.net = nh.DynamicLinearSequential(input_dim, output_dim, 
                            nh.parse_layers(layers), nh.parse_functions(activ_function), 
                            nh.get_attrs(output_function))
    def forward(self, x):
        return self.net(x)