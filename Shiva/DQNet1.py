import torch
import torch.nn as nn
import numpy as np
from Network_Templates.AbstractNetwork import Network


class DQNet(Network):
    def __init__(self, input_size, output_size):
        super(DQNet, self).__init__(input_size, output_size)
        self.net = nn.Sequential(
            nn.Linear(input_size,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,output_size)
            )

    def forward(self,x):
        return self.net(x)
        self.net = nn.Sequential(
            nn.Linear(input_size,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,output_size)
            )

    def forward(self,x):
        return self.net(x)