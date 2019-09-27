import torch
import numpy as np

class Network(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def forward():
        pass

class DQNet(Network):
    
    def __init__(self, input_size, hidden_layer1, hidden_layer2, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_layer1)
        self.linear2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.linear3 = nn.Linear(hidden_layer2, output_size)
        
    def forward(self,x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        return x
   
