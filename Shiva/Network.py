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

class DQNet(Network):
    def __init__(self, input_size, HIDDEN_SIZE_1, HIDDEN_SIZE_2, output_size):
        super(DQNet, self).__init__(input_size, output_size)
        self.net = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, output_size),
        )
        
    def forward(self, x):
        return self.net(x)
   
