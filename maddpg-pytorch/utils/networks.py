import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from .i2a import RolloutEncoder
import numpy as np
from utils.misc import processor
from .misc import onehot_from_logits
class MLPNetwork_Actor(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=int(1024), nonlin=F.relu, norm_in=True, discrete_action=True,agent=object,maddpg=object):
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
        if self.agent.device == 'cuda':
            self.cast = lambda x: x.to(maddpg.torch_device)
        else:
            self.cast = lambda x: x.cpu()
        


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
        h1 = self.nonlin(self.fc1(self.cast(X)))
        h2 = self.nonlin(self.fc2(h1))
        h3 = self.nonlin(self.fc3(h2))
        h4 = self.nonlin(self.fc4(h3))

        if not self.discrete_action:
            self.final_out_action = self.out_action_fn(self.out_action(h4))
            self.final_out_params = self.out_param_fn(self.out_param(h4))
            if self.final_out_action.shape[0] == 3:
                out = np.asarray(torch.cat((self.final_out_action, self.final_out_params)).cpu().data.numpy())
            else:
                out = torch.cat((self.final_out_action, self.final_out_params),1)
            #if self.count % 100 == 0:
            #    print(out)
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
    def __init__(self, input_dim, out_dim, hidden_dim=int(1024), nonlin=F.relu, norm_in=True, agent=object,n_atoms=51,D4PG=False,TD3=False,maddpg=None):
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
       
        if self.agent.device == 'cuda':
            self.cast = lambda x: x.to(maddpg.torch_device)
        else:
            self.cast = lambda x: x.cpu()
              

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
        h1 = self.nonlin(self.fc1(self.cast(X)))
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
        h1 = self.nonlin(self.fc1(self.cast(X)))
        h2 = self.nonlin(self.fc2(h1))
        h3 = self.nonlin(self.fc3(h2))
        h4 = self.nonlin(self.fc4(h3))
        
        if self.TD3:
            Q2_h1 = self.nonlin(self.Q2_fc1(self.cast(X)))
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

class LSTMNetwork_Critic(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=int(512), nonlin=F.relu, agent=object,n_atoms=51,D4PG=False,TD3=False,maddpg=object):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (intT): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(LSTMNetwork_Critic, self).__init__()

        self.agent=agent
        self.action_size = 3
        self.param_size = 5
        self.count = 0
        self.TD3 = TD3
        self.batch_size = agent.batch_size
        self.torch_device = maddpg.torch_device
        self.hidden_dim_lstm = agent.hidden_dim_lstm
        if agent.device == 'cuda':
            self.hidden_tuple = (Variable(torch.zeros(1, self.batch_size, self.hidden_dim_lstm)).to(self.torch_device),
                                    Variable(torch.zeros(1, self.batch_size, self.hidden_dim_lstm)).to(self.torch_device))
            self.hidden_tuple2 = (Variable(torch.zeros(1, self.batch_size, self.hidden_dim_lstm)).to(self.torch_device),
                                    Variable(torch.zeros(1, self.batch_size, self.hidden_dim_lstm)).to(self.torch_device))
        else:
            self.hidden_tuple = (Variable(torch.zeros(1, self.batch_size, self.hidden_dim_lstm)),
                                    Variable(torch.zeros(1, self.batch_size, self.hidden_dim_lstm)))
            self.hidden_tuple2 = (Variable(torch.zeros(1, self.batch_size, self.hidden_dim_lstm)),
                                    Variable(torch.zeros(1, self.batch_size, self.hidden_dim_lstm)))

        if D4PG:
            self.out_dim = n_atoms
        else:
            self.out_dim = 1
            
        self.maddpg=maddpg
        if self.agent.device == 'cuda':
            self.cast = lambda x: x.to(self.torch_device)
        else:
            self.cast = lambda x: x.cpu()
              
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc2 = nn.Linear(512, 256)
        self.fc2.weight.data.normal_(0, 0.01) 
        self.lstm3 = nn.LSTM(256,self.hidden_dim_lstm)
        
        if TD3: # second critic
            self.Q2_fc1 = nn.Linear(input_dim, 512)
            self.Q2_fc1.weight.data.normal_(0, 0.01)
            self.Q2_fc2 = nn.Linear(512, 256)
            self.Q2_fc2.weight.data.normal_(0, 0.01)
            self.Q2_lstm3 = nn.LSTM(256,self.hidden_dim_lstm)
            
        self.out = nn.Linear(self.hidden_dim_lstm, self.out_dim)
        self.register_buffer("supports",torch.arange(agent.vmin,agent.vmax + agent.delta, agent.delta))
            
        if TD3: # second critic
            self.Q2_out = nn.Linear(self.hidden_dim_lstm, self.out_dim)

        self.nonlin = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.out_fn = lambda x: x
    
    def init_hidden(self, batch_size,torch_device):
        self.torch_device = self.maddpg.torch_device

        if self.agent.device == 'cuda':
            self.hidden_tuple = (Variable(torch.zeros(1, batch_size, self.hidden_dim_lstm)).to(self.torch_device),
                                            Variable(torch.zeros(1, batch_size, self.hidden_dim_lstm)).to(self.torch_device))
            self.hidden_tuple2 = (Variable(torch.zeros(1, batch_size, self.hidden_dim_lstm)).to(self.torch_device),
                                            Variable(torch.zeros(1, batch_size, self.hidden_dim_lstm)).to(self.torch_device))
        else:
            self.hidden_tuple = (Variable(torch.zeros(1, batch_size, self.hidden_dim_lstm)),
                                            Variable(torch.zeros(1, batch_size, self.hidden_dim_lstm)))
            self.hidden_tuple2 = (Variable(torch.zeros(1, batch_size, self.hidden_dim_lstm)),
                                            Variable(torch.zeros(1, batch_size, self.hidden_dim_lstm)))
    
    def set_hidden(self, h1, h2):
        self.hidden_tuple = h1
        self.hidden_tuple2 = h2

    def Q1(self, X):
        """
        This function is for the TD3 functionality to use only a single critic for the policy update.
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of critic Q1
        """
        h1 = self.nonlin(self.fc1(self.cast(X)))
        h2 = self.nonlin(self.fc2(h1))
        h3, self.hidden_tuple = self.lstm3(h2, self.hidden_tuple)
        out = self.out_fn(self.out(h3))
        return out

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.cast(X)))
        h2 = self.nonlin(self.fc2(h1))
        h3, self.hidden_tuple = self.lstm3(h2, self.hidden_tuple)
        
        if self.TD3:
            Q2_h1 = self.nonlin(self.Q2_fc1(self.cast(X)))
            Q2_h2 = self.nonlin(self.Q2_fc2(Q2_h1))
            Q2_h3, self.hidden_tuple2 = self.Q2_lstm3(Q2_h2, self.hidden_tuple2)
        
        out = self.out_fn(self.out(h3))
        if self.TD3: # 2nd critic
            Q2_out = self.out_fn(self.Q2_out(Q2_h3))
            return out, Q2_out, self.hidden_tuple, self.hidden_tuple2
        return out

    def distr_to_q(self, distr):
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)

    
class I2A_Network(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, EM_out_dim, hidden_dim=int(1024), nonlin=F.relu, norm_in=True, discrete_action=True,agent=object,I2A=False,rollout_steps=5,EM=object,pol_prime=object,imagined_pol=object,LSTM_hidden=64,maddpg=object):
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
        self.n_branches = self.agent.n_branches
        self.rollout_steps = rollout_steps

        self.I2A = I2A
        self.LSTM = maddpg.LSTM
        self.encoder = RolloutEncoder(EM_out_dim,hidden_size=LSTM_hidden)

        # save refs without registering
        object.__setattr__(self, "EM", EM)
        object.__setattr__(self, "pol_prime", pol_prime)
        object.__setattr__(self, "imagined_pol",imagined_pol)

        
        if self.agent.device == 'cuda':
            self.cast = lambda x: x.to(maddpg.torch_device)
        else:
            self.cast = lambda x: x.cpu()
        self.norm_in = norm_in
        if I2A:
            self.fc1 = nn.Linear(input_dim + LSTM_hidden * self.n_branches, 1024)
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

        if not self.I2A:     
            h1 = self.nonlin(self.fc1(self.cast(X)))
            h2 = self.nonlin(self.fc2(h1))
            h3 = self.nonlin(self.fc3(h2))
            h4 = self.nonlin(self.fc4(h3))

            if not self.discrete_action:
                self.final_out_action = self.out_action_fn(self.out_action(h4))
                self.final_out_params = self.out_param_fn(self.out_param(h4))

                if len(self.final_out_action.shape) == 2:
                    out = torch.cat((self.final_out_action,self.final_out_params),1)
                else:
                    out = torch.cat((self.final_out_action, self.final_out_params),2)
                #if self.count % 100 == 0:
                #    print(out)
                self.count += 1
        else: # I2A

            batch_size = X[0].size()[0]
            fx = [self.cast(x).float() for x in X]

            enc_rollouts = self.rollouts_batch(fx)
            if not self.LSTM:
                fc_in = [torch.cat((f,e),dim=1) for f,e in zip(fx,enc_rollouts)]
            else:
                fc_in = [torch.cat((f,e),dim=2) for f,e in zip(fx,enc_rollouts)]
            #     fc_in = [torch.cat((f,e),dim=2) for f,e in zip(fx,enc_rollouts)]
            out = [self.ag_forward(x) for x in fc_in]
        return out

    def ag_forward(self,x):
        h1 = self.nonlin(self.fc1(x))
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
        # batch is a list of obs for each agent
        batch_size = batch[0].size()[0] # agent 1's obs tensor batch size
        batch_rest = batch[0].size()[1:]
        if batch_size == 1:
            obs_batch_v = [b.expand(batch_size * self.n_branches, *batch_rest) for b in batch]
        else:
            obs_batch_v = [b.unsqueeze(1) for b in batch]
            obs_batch_v = [o.expand(batch_size, self.n_branches, *batch_rest) for o in obs_batch_v]
            obs_batch_v = [o.contiguous().view(-1, *batch_rest) for o in obs_batch_v]
        #actions = np.tile(np.arange(0, self.n_branches, dtype=np.int64), batch_size)
        actions = []
        if self.agent.imagination_policy_branch: # needs revision
            traj_1 = [self.pol_prime(b) for b in batch]
            traj_2 = [self.pol_imagined_pol(b) for b in batch]
            actions_1 = [torch.cat(
                    (onehot_from_logits(out[:,:self.agent.action_dim]),out[:,self.agent.action_dim:]),dim=1) for out in traj_1]
            actions_2 = [torch.cat(
                    (onehot_from_logits(out[:,:self.agent.action_dim]),out[:,self.agent.action_dim:]),dim=1) for out in traj_2]
            actions = [torch.stack((t1.view(batch_size,-1),t2.view(batch_size,-1)),dim=1).view(self.n_branches*batch_size,-1) for t1,t2 in zip(actions_1,actions_2)] # use our agent and another pretrained policy as a branch
        else:
            prime_out = [self.pol_prime(b) for b in batch] # use all agents actions for rollout
            actions = [torch.cat(
                    (onehot_from_logits(out[:,:self.agent.action_dim]),out[:,self.agent.action_dim:]),dim=1) for out in prime_out]
        step_obs, step_rewards,step_ws = [], [],[]
        for step_idx in range(self.rollout_steps):
            #actions = processor(torch.tensor(actions).float(),device=self.agent.device,torch_device=self.agent.maddpg.torch_device)
            acs_flat = torch.cat(actions,dim=1)
            obs_next_v, reward_v, ws_v = [], [], []
            for ag in range(self.agent.maddpg.nagents_team):
                obs, rew,ws = self.EM(torch.cat((obs_batch_v[ag], acs_flat),dim=1))
                obs_next_v.append(obs.detach())
                reward_v.append(rew.detach())
                ws_v.append(ws.detach())

            step_obs.append(obs_next_v)
            step_rewards.append(reward_v)
            step_ws.append(ws_v)

            # don't need actions for the last step
            if step_idx == self.rollout_steps-1:
                break
            # combine the delta from EM into new observation
            obs_batch_v = [curr + delta for curr, delta in zip(obs_batch_v,obs_next_v)]
            # select actions
        if self.agent.imagination_policy_branch: # needs revision
            traj_1 = [self.pol_prime(b) for b in obs_batch_v]
            traj_2 = [self.pol_imagined_pol(b) for b in obs_batch_v]
            actions_1 = [torch.cat(
                    (onehot_from_logits(out[:,:self.agent.action_dim]),out[:,self.agent.action_dim:]),dim=1) for out in traj_1]
            actions_2 = [torch.cat(
                    (onehot_from_logits(out[:,:self.agent.action_dim]),out[:,self.agent.action_dim:]),dim=1) for out in traj_2]
            actions = [torch.stack((t1.view(batch_size,-1),t2.view(batch_size,-1)),dim=1).view(self.n_branches*batch_size,-1) for t1,t2 in zip(actions_1,actions_2)] # use our agent and another pretrained policy as a branch
        else:
            prime_out = [self.pol_prime(b) for b in obs_batch_v] # use all agents actions for rollout
            actions = [torch.cat(
                    (onehot_from_logits(out[:,:self.agent.action_dim]),out[:,self.agent.action_dim:]),dim=1) for out in prime_out]
        ag_obs = list(zip(*step_obs))
        ag_rews = list(zip(*step_rewards))
        ag_ws = list(zip(*step_ws))
        
        step_obs_v = [torch.stack(ao) for ao in ag_obs]

        step_rewards_v = [torch.stack(ar) for ar in ag_rews]
        step_ws_v = [torch.stack(aws) for aws in ag_ws]
        flat_enc_v = [self.encoder(so, sr,sws).view(batch_size,-1) for so,sr,sws in zip(step_obs_v,step_rewards_v,step_ws_v)]
        return flat_enc_v
    
    def distr_to_q(self, distr):
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)

    
    
class Dimension_Reducer(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, nonlin=F.elu, norm_in=True, agent=object,maddpg=None):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (intT): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(Dimension_Reducer, self).__init__()

        self.agent = agent
        if self.agent.device == 'cuda':
            self.cast = lambda x: x.to(maddpg.torch_device)
        else:
            self.cast = lambda x: x.cpu()
              
        
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc1.weight.data.normal_(0, 0.01) 
        self.fc2 = nn.Linear(32, 16)
        self.fc2.weight.data.normal_(0, 0.01) 
        self.fc3 = nn.Linear(16, 32)
        self.fc3.weight.data.normal_(0, 0.01) 
        self.out = nn.Linear(32+(3*input_dim),input_dim)
        self.out.weight.data.normal_(0, 0.01) 
        self.nonlin = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.out_fn = lambda x: x


    def reduce(self, X):
        """
        This function is for the TD3 functionality to use only a single critic for the policy update.
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of critic Q1
        """
        h1 = self.nonlin(self.fc1(self.cast(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(h2) # Should there be linear or relu on output layer
        return out

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        batch = len(X)
        max_v = X.max(dim=0)[0].repeat(batch,1)
        min_v = X.min(dim=0)[0].repeat(batch,1)
        avg_v = X.mean(dim=0).repeat(batch,1)
        stats = torch.cat((max_v,min_v,avg_v),dim=1)
        h1 = self.nonlin(self.fc1(self.cast(X)))
        h2 = self.nonlin(self.fc2(h1))
        h3 = self.nonlin(self.fc3(h2))
        h3_cat = torch.cat((h3,stats),dim=1)
        out = self.out_fn(self.out(h3_cat))
        return out

class LSTM_Actor(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, EM_out_dim, hidden_dim=int(512), nonlin=F.relu, norm_in=False, discrete_action=True,agent=object,I2A=False,rollout_steps=5,EM=object,pol_prime=object,imagined_pol=object,LSTM_hidden=64,maddpg=object):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (intT): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(LSTM_Actor, self).__init__()

        self.agent=agent
        self.discrete_action = discrete_action
        self.action_size = 3
        self.param_size = 5
        self.count = 0
        self.n_branches = self.agent.n_branches
        self.rollout_steps = rollout_steps
        self.maddpg = maddpg
        self.torch_device = maddpg.torch_device
        self.hidden_dim_lstm = agent.hidden_dim_lstm
        self.batch_size = agent.batch_size
        self.I2A = I2A
        self.LSTM = maddpg.LSTM
        self.encoder = RolloutEncoder(EM_out_dim,hidden_size=LSTM_hidden)

        if agent.device == 'cuda':
            self.hidden_tuple = (Variable(torch.zeros(1, self.batch_size, self.hidden_dim_lstm)).to(self.torch_device),
                                    Variable(torch.zeros(1, self.batch_size, self.hidden_dim_lstm)).to(self.torch_device))
        else:
            self.hidden_tuple = (Variable(torch.zeros(1, self.batch_size, self.hidden_dim_lstm)),
                                    Variable(torch.zeros(1, self.batch_size, self.hidden_dim_lstm)))

        # save refs without registering
        object.__setattr__(self, "EM", EM)
        object.__setattr__(self, "pol_prime", pol_prime)
        object.__setattr__(self, "imagined_pol",imagined_pol)

        
        if self.agent.device == 'cuda':
            self.cast = lambda x: x.to(maddpg.torch_device)
        else:
            self.cast = lambda x: x.cpu()
        self.norm_in = norm_in
        if I2A:
            self.fc1 = nn.Linear(input_dim + LSTM_hidden * self.n_branches, 512)
        else:
            self.fc1 = nn.Linear(input_dim, 512)

        
        self.fc1.weight.data.normal_(0, 0.01) 
        self.fc2 = nn.Linear(512, 256)
        self.fc2.weight.data.normal_(0, 0.01) 
        self.fc3 = nn.Linear(256, 256)
        self.fc3.weight.data.normal_(0, 0.01) 
        self.lstm4 = nn.LSTM(256, self.hidden_dim_lstm)

        # hard coded values
        self.out_action = nn.Linear(self.hidden_dim_lstm, self.action_size)
        self.out_action.weight.data.normal_(0, 0.01) 

        self.out_param = nn.Linear(self.hidden_dim_lstm, self.param_size)
        self.out_param.weight.data.normal_(0, 0.01) 
        self.out_param_fn = lambda x: x
        self.out_action_fn = lambda x: x

        self.nonlin = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.out_fn = lambda x: x

    def init_hidden(self, batch_size):
        if self.agent.device == 'cuda':
            self.hidden_tuple = (Variable(torch.zeros(1, batch_size, self.hidden_dim_lstm)).to(self.torch_device),
                                            Variable(torch.zeros(1, batch_size, self.hidden_dim_lstm)).to(self.torch_device))
        else:
            self.hidden_tuple = (Variable(torch.zeros(1, batch_size, self.hidden_dim_lstm)),
                                            Variable(torch.zeros(1, batch_size, self.hidden_dim_lstm)))

    def set_hidden(self, h1):
        self.hidden_tuple = h1

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """

        if not self.I2A:     
            if X.shape[0] == 1:
                X = torch.unsqueeze(X,dim=0)
            h1 = self.nonlin(self.fc1(self.cast(X)))
            h2 = self.nonlin(self.fc2(h1))
            h3 = self.nonlin(self.fc3(h2))
            h4, self.hidden_tuple = self.lstm4(h3, self.hidden_tuple)

            if not self.discrete_action:
                self.final_out_action = self.out_action_fn(self.out_action(h4))
                self.final_out_params = self.out_param_fn(self.out_param(h4))

                out = torch.cat((self.final_out_action, self.final_out_params),2)
                #if self.count % 100 == 0:
                #    print(out)
                self.count += 1
            return out
        else: # I2A

            batch_size = X[0].size()[0]
            fx = [self.cast(x).float() for x in X]

            enc_rollouts = self.rollouts_batch(fx)
            if not self.LSTM:
                fc_in = [torch.cat((f,e),dim=1) for f,e in zip(fx,enc_rollouts)]
            else:
                fc_in = [torch.cat((f,e),dim=2) for f,e in zip(fx,enc_rollouts)]
            #     fc_in = [torch.cat((f,e),dim=2) for f,e in zip(fx,enc_rollouts)]
            out = []
            ht = []
            for x in fc_in:
                o,h = self.ag_forward(x)
                out.append(o)
                ht.append(h) 
            return out, ht


    def ag_forward(self,x):
        h1 = self.nonlin(self.fc1(x))
        h2 = self.nonlin(self.fc2(h1))
        h3 = self.nonlin(self.fc3(h2))
        h4,self.hidden_tuple = self.lstm4(h3,self.hidden_tuple)

        if not self.discrete_action:
            self.final_out_action = self.out_action_fn(self.out_action(h4))
            self.final_out_params = self.out_param_fn(self.out_param(h4))

            out = torch.cat((self.final_out_action, self.final_out_params),2)
            #if self.count % 100 == 0:
            #    print(out)
            self.count += 1
        return out , self.hidden_tuple

    def rollouts_batch(self, batch):
        # batch is a list of obs for each agent
        batch_size = batch[0].size()[0] # agent 1's obs tensor batch size
        batch_rest = batch[0].size()[1:]
        if batch_size == 1:
            obs_batch_v = [b.expand(batch_size * self.n_branches, *batch_rest) for b in batch]
        else:
            obs_batch_v = [b.unsqueeze(1) for b in batch]
            obs_batch_v = [o.expand(batch_size, self.n_branches, *batch_rest) for o in obs_batch_v]
            obs_batch_v = [o.contiguous().view(-1, *batch_rest) for o in obs_batch_v]
        #actions = np.tile(np.arange(0, self.n_branches, dtype=np.int64), batch_size)
        actions = []
        if self.agent.imagination_policy_branch: # needs revision
            traj_1 = [self.pol_prime(b) for b in batch]
            traj_2 = [self.pol_imagined_pol(b) for b in batch]
            actions_1 = [torch.cat(
                    (onehot_from_logits(out[:,:self.agent.action_dim]),out[:,self.agent.action_dim:]),dim=1) for out in traj_1]
            actions_2 = [torch.cat(
                    (onehot_from_logits(out[:,:self.agent.action_dim]),out[:,self.agent.action_dim:]),dim=1) for out in traj_2]
            actions = [torch.stack((t1.view(batch_size,-1),t2.view(batch_size,-1)),dim=1).view(self.n_branches*batch_size,-1) for t1,t2 in zip(actions_1,actions_2)] # use our agent and another pretrained policy as a branch
        else:
            prime_out = [self.pol_prime(b) for b in batch] # use all agents actions for rollout
            actions = [torch.cat(
                    (onehot_from_logits(out[:,:self.agent.action_dim]),out[:,self.agent.action_dim:]),dim=1) for out in prime_out]
        step_obs, step_rewards,step_ws = [], [],[]
        for step_idx in range(self.rollout_steps):
            #actions = processor(torch.tensor(actions).float(),device=self.agent.device,torch_device=self.agent.maddpg.torch_device)
            acs_flat = torch.cat(actions,dim=1)
            obs_next_v, reward_v, ws_v = [], [], []
            for ag in range(self.agent.maddpg.nagents_team):
                obs, rew,ws = self.EM(torch.cat((obs_batch_v[ag], acs_flat),dim=1))
                obs_next_v.append(obs.detach())
                reward_v.append(rew.detach())
                ws_v.append(ws.detach())

            step_obs.append(obs_next_v)
            step_rewards.append(reward_v)
            step_ws.append(ws_v)

            # don't need actions for the last step
            if step_idx == self.rollout_steps-1:
                break
            # combine the delta from EM into new observation
            obs_batch_v = [curr + delta for curr, delta in zip(obs_batch_v,obs_next_v)]
            # select actions
        if self.agent.imagination_policy_branch: # needs revision
            traj_1 = [self.pol_prime(b) for b in obs_batch_v]
            traj_2 = [self.pol_imagined_pol(b) for b in obs_batch_v]
            actions_1 = [torch.cat(
                    (onehot_from_logits(out[:,:self.agent.action_dim]),out[:,self.agent.action_dim:]),dim=1) for out in traj_1]
            actions_2 = [torch.cat(
                    (onehot_from_logits(out[:,:self.agent.action_dim]),out[:,self.agent.action_dim:]),dim=1) for out in traj_2]
            actions = [torch.stack((t1.view(batch_size,-1),t2.view(batch_size,-1)),dim=1).view(self.n_branches*batch_size,-1) for t1,t2 in zip(actions_1,actions_2)] # use our agent and another pretrained policy as a branch
        else:
            prime_out = [self.pol_prime(b) for b in obs_batch_v] # use all agents actions for rollout
            actions = [torch.cat(
                    (onehot_from_logits(out[:,:self.agent.action_dim]),out[:,self.agent.action_dim:]),dim=1) for out in prime_out]
        ag_obs = list(zip(*step_obs))
        ag_rews = list(zip(*step_rewards))
        ag_ws = list(zip(*step_ws))
        
        step_obs_v = [torch.stack(ao) for ao in ag_obs]

        step_rewards_v = [torch.stack(ar) for ar in ag_rews]
        step_ws_v = [torch.stack(aws) for aws in ag_ws]
        flat_enc_v = [self.encoder(so, sr,sws).view(batch_size,-1) for so,sr,sws in zip(step_obs_v,step_rewards_v,step_ws_v)]
        return flat_enc_v
    
    def distr_to_q(self, distr):
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)
