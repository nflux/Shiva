import torch
import torch.nn as nn
import numpy as np
from Network_Templates.AbstractNetwork import Network


class DQNet(Network):
    def __init__(self, input_size, output_size):
        super(DQNet, self).__init__(input_size, output_size)