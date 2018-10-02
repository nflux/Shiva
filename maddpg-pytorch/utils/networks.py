import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=int(1024), nonlin=F.relu, norm_in=True, discrete_action=True, is_actor=False,agent=object):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (intT): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        self.agent=agent
        self.is_actor = is_actor
        self.discrete_action = discrete_action
        self.action_size = 3
        self.param_size = 5
        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, 1024)
        
        self.fc1.weight.data.normal_(0, 0.01) 
        self.fc2 = nn.Linear(1024, 512)
        self.fc2.weight.data.normal_(0, 0.01) 
        self.fc3 = nn.Linear(512, 256)
        self.fc3.weight.data.normal_(0, 0.01) 
        self.fc4 = nn.Linear(256, 128)
        self.fc4.weight.data.normal_(0, 0.01) 
        if is_actor:
            # hard coded values
            self.out_action = nn.Linear(128, self.action_size)
            self.out_action.weight.data.normal_(0, 0.01) 

            self.out_param = nn.Linear(128, self.param_size)
            self.out_param.weight.data.normal_(0, 0.01) 
            #self.out_param_fn = F.tanh
            self.out_param_fn = lambda x: x
        else:
            self.out = nn.Linear(128,out_dim)
            self.out.weight.data.normal_(0, 0.01) 

        

        self.nonlin = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        h3 = self.nonlin(self.fc3(h2))
        h4 = self.nonlin(self.fc4(h3))
        
        
        if self.is_actor and not self.discrete_action:
 
            self.final_out_action = self.out_fn(self.out_action(h4))
            self.final_out_params = Variable(self.out_param_fn(self.out_param(h4)),requires_grad=True)
            #h = self.current_params.register_hook(self.invert)
            out = torch.cat((self.final_out_action, self.final_out_params),1)
           # print("OUT",out)

        else:
            out = self.out_fn(self.out(h4))

        return out
