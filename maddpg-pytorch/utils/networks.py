import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=int(1024), nonlin=F.relu, norm_in=True, discrete_action=True, is_actor=False,agent=object,n_atoms=51,D4PG=False,TD3=False):
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
        self.count = 0
        self.TD3 = TD3
        if D4PG:
            self.out_dim = n_atoms
        else:
            self.out_dim = 1
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
        
        if TD3: # second critic
            self.Q2_fc1 = nn.Linear(input_dim, 1024)
            self.Q2_fc1.weight.data.normal_(0, 0.01) 
            self.Q2_fc2 = nn.Linear(1024, 512)
            self.Q2_fc2.weight.data.normal_(0, 0.01) 
            self.Q2_fc3 = nn.Linear(512, 256)
            self.Q2_fc3.weight.data.normal_(0, 0.01) 
            self.Q2_fc4 = nn.Linear(256, 128)
            self.Q2_fc4.weight.data.normal_(0, 0.01) 
            
        if is_actor:
            # hard coded values
            self.out_action = nn.Linear(128, self.action_size)
            self.out_action.weight.data.normal_(0, 0.01) 

            self.out_param = nn.Linear(128, self.param_size)
            self.out_param.weight.data.normal_(0, 0.01) 
            self.out_param_fn = lambda x: x
            #self.out_action_fn = F.softmax
            #self.out_action_fn = F.log_softmax
            self.out_action_fn = lambda x: x

        else: # is critic
            self.out = nn.Linear(128,self.out_dim)
            self.register_buffer("supports",torch.arange(agent.vmin,agent.vmax + agent.delta, agent.delta))
            if TD3: # second critic
                self.Q2_out = nn.Linear(128,self.out_dim) 
                self.Q2_out.weight.data.normal_(0,0.01)
        
            self.out.weight.data.normal_(0, 0.01)
            
        self.nonlin = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.out_fn = lambda x: x


    def Q1(self, X):
        """
        This function is for the TD3 functionality to use only a single critic for the policy update.
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of critic Q1
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        h3 = self.nonlin(self.fc3(h2))
        h4 = self.nonlin(self.fc4(h3))
        out = self.out_fn(self.out(h4))
        return out

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
        
        if not self.is_actor and self.TD3:
            Q2_h1 = self.nonlin(self.Q2_fc1(self.in_fn(X)))
            Q2_h2 = self.nonlin(self.Q2_fc2(Q2_h1))
            Q2_h3 = self.nonlin(self.Q2_fc3(Q2_h2))
            Q2_h4 = self.nonlin(self.Q2_fc4(Q2_h3))



        if self.is_actor and not self.discrete_action:
            self.final_out_action = self.out_action_fn(self.out_action(h4))
            self.final_out_params = self.out_param_fn(self.out_param(h4))
            out = torch.cat((self.final_out_action, self.final_out_params),1)
            #if self.count % 100 == 0:
            #    print(out)
            self.count += 1
        else: # critic
            out = self.out_fn(self.out(h4))
            if self.TD3: # 2nd critic
                Q2_out = self.out_fn(self.Q2_out(Q2_h4))
                return out,Q2_out
        return out

    def distr_to_q(self, distr):
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)
