import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from .i2a import RolloutEncoder
import numpy as np
class MLPNetwork_Actor(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=int(1024), nonlin=F.relu, norm_in=True, discrete_action=True,agent=object):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (intT): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork_Actor, self).__init__()

        self.agent=agent
        self.discrete_action = discrete_action
        self.action_size = 3
        self.param_size = 5
        self.count = 0

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

        # hard coded values
        self.out_action = nn.Linear(128, self.action_size)
        self.out_action.weight.data.normal_(0, 0.01) 

        self.out_param = nn.Linear(128, self.param_size)
        self.out_param.weight.data.normal_(0, 0.01) 
        self.out_param_fn = lambda x: x
        #self.out_action_fn = F.softmax
        #self.out_action_fn = F.log_softmax
        self.out_action_fn = lambda x: x

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

        if not self.discrete_action:
            self.final_out_action = self.out_action_fn(self.out_action(h4))
            self.final_out_params = self.out_param_fn(self.out_param(h4))
            if self.final_out_action.shape[0] == 3:
                out = np.asarray(torch.cat((self.final_out_action, self.final_out_params)).data.numpy())
            else:
                out = torch.cat((self.final_out_action, self.final_out_params),1)
            if self.count % 100 == 0:
                print(out)
            self.count += 1

        return out

    def distr_to_q(self, distr):
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)

    
class MLPNetwork_Critic(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=int(1024), nonlin=F.relu, norm_in=True, agent=object,n_atoms=51,D4PG=False,TD3=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (intT): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork_Critic, self).__init__()

        self.agent=agent
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
        
        if self.TD3:
            Q2_h1 = self.nonlin(self.Q2_fc1(self.in_fn(X)))
            Q2_h2 = self.nonlin(self.Q2_fc2(Q2_h1))
            Q2_h3 = self.nonlin(self.Q2_fc3(Q2_h2))
            Q2_h4 = self.nonlin(self.Q2_fc4(Q2_h3))
            
        out = self.out_fn(self.out(h4))
        if self.TD3: # 2nd critic
            Q2_out = self.out_fn(self.Q2_out(Q2_h4))
            return out,Q2_out
        return out

    def distr_to_q(self, distr):
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)

    
class I2A_Network(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, EM_out_dim, hidden_dim=int(1024), nonlin=F.relu, norm_in=True, discrete_action=True,agent=object,I2A=False,rollout_steps=5,EM=object,pol_prime=object,LSTM_hidden=64):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (intT): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(I2A_Network, self).__init__()

        self.agent=agent
        self.discrete_action = discrete_action
        self.action_size = 3
        self.param_size = 5
        self.count = 0
        self.n_actions = self.agent.n_actions
        self.rollout_steps = rollout_steps
        ROLLOUT_HIDDEN = 64 # ??

        self.I2A = I2A
        self.encoder = RolloutEncoder(EM_out_dim,hidden_size=ROLLOUT_HIDDEN)

        # save refs without registering
        object.__setattr__(self, "EM", EM)
        object.__setattr__(self, "pol_prime", pol_prime)

        
        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
            
        if I2A:
            self.fc1 = nn.Linear(input_dim + ROLLOUT_HIDDEN * self.n_actions, 1024)
        else:
            self.fc1 = nn.Linear(input_dim, 1024)

        
        self.fc1.weight.data.normal_(0, 0.01) 
        self.fc2 = nn.Linear(1024, 512)
        self.fc2.weight.data.normal_(0, 0.01) 
        self.fc3 = nn.Linear(512, 256)
        self.fc3.weight.data.normal_(0, 0.01) 
        self.fc4 = nn.Linear(256, 128)
        self.fc4.weight.data.normal_(0, 0.01) 

        # hard coded values
        self.out_action = nn.Linear(128, self.action_size)
        self.out_action.weight.data.normal_(0, 0.01) 

        self.out_param = nn.Linear(128, self.param_size)
        self.out_param.weight.data.normal_(0, 0.01) 
        self.out_param_fn = lambda x: x
        #self.out_action_fn = F.softmax
        #self.out_action_fn = F.log_softmax
        self.out_action_fn = lambda x: x

        self.nonlin = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        
        if self.I2A:
            fx = self.in_fn(X).float()
            enc_rollouts = self.rollouts_batch(fx)
            fc_in = torch.cat((fx,enc_rollouts),dim=1)
            h1 = self.nonlin(self.fc1(fc_in))
        else:
            h1 = self.nonlin(self.fc1(self.in_fn(X)))

        h2 = self.nonlin(self.fc2(h1))
        h3 = self.nonlin(self.fc3(h2))
        h4 = self.nonlin(self.fc4(h3))

        if not self.discrete_action:
            self.final_out_action = self.out_action_fn(self.out_action(h4))
            self.final_out_params = self.out_param_fn(self.out_param(h4))
            out = torch.cat((self.final_out_action, self.final_out_params),1)
            #if self.count % 100 == 0:
            #    print(out)
            self.count += 1

        return out

    def rollouts_batch(self, batch):
        batch_size = batch.size()[0]
        batch_rest = batch.size()[1:]
        if batch_size == 1:
            obs_batch_v = batch.expand(batch_size * self.n_actions, *batch_rest)
        else:
            obs_batch_v = batch.unsqueeze(1)
            obs_batch_v = obs_batch_v.expand(batch_size, self.n_actions, *batch_rest)
            obs_batch_v = obs_batch_v.contiguous().view(-1, *batch_rest)
        #actions = np.tile(np.arange(0, self.n_actions, dtype=np.int64), batch_size)
        actions = []
        for item in batch:
            actions.append(self.pol_prime(item))
            actions.append(np.array([1.0,0.0,0.0,np.random.uniform(-1,1),np.random.uniform(-1,1),0.0,0.0,0.0]))
            actions.append(np.array([0.0,1.0,0.0,0.0,0.0,np.random.uniform(-1,1),0.0,0.0]))
            actions.append(np.array([0.0,0.0,1.0,0.0,0.0,0.0,np.random.uniform(-1,1),np.random.uniform(-1,1)]))
                           
        step_obs, step_rewards,step_ws = [], [],[]
        for step_idx in range(self.rollout_steps):
            actions_t = torch.tensor(actions).float()
            #EM_in = torch.cat((*obs_batch_v, *actions_t),dim=1)

            obs_next_v, reward_v,ws_v = self.EM(torch.cat((obs_batch_v, actions_t),dim=1))                

            step_obs.append(obs_next_v.detach())
            step_rewards.append(reward_v.detach())
            step_ws.append(ws_v.detach())
            # don't need actions for the last step
            if step_idx == self.rollout_steps-1:
                break
            # combine the delta from EM into new observation
            obs_batch_v = obs_batch_v + obs_next_v
            # select actions
            actions = self.pol_prime(obs_batch_v)
        step_obs_v = torch.stack(step_obs)
        step_rewards_v = torch.stack(step_rewards)
        step_ws_v = torch.stack(step_ws)
        flat_enc_v = self.encoder(step_obs_v, step_rewards_v,step_ws_v)
        return flat_enc_v.view(batch_size, -1)  
    
    def distr_to_q(self, distr):
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)
