import torch
import torch.nn as nn
import numpy as np

class Network(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Network, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def forward(self):
        pass
