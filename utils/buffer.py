import numpy as np
from torch import Tensor,squeeze
from torch.autograd import Variable
import operator
import torch
import math

def roll(tensor, rollover):
    '''
    Roll over the first axis of a tensor
    '''
    return torch.cat((tensor[-rollover:], tensor[:-rollover]))

def roll2(tensor, rollover):
    '''
    Roll over the second axis of a tensor
    '''
    return torch.cat((tensor[:,-rollover:], tensor[:,:-rollover]), dim=1)


class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, num_agents, obs_dim, ac_dim, hidden_dim_lstm,k,prox_item_size,SIL,pretrain=False):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        # self.max_episodes = int(max_steps/episode_length)
        self.num_agents = num_agents

        self.obs_dim = obs_dim
        self.ac_dim = ac_dim  
        self.obs_acs_dim = obs_dim + ac_dim 
        self.hidden_dim_lstm = hidden_dim_lstm 
        self.prox_item_size = prox_item_size   
        self.pretrain = pretrain
        self.total_dim = self.getTotalDim(pretrain, obs_dim, ac_dim,k,hidden_dim_lstm,prox_item_size)

        self.k = k
        self.SIL = SIL

        # for _ in range(num_agents):
        #     self.obs_buffs.append(torch.zeros((max_steps, obs_dim)))
        #     self.ac_buffs.append(torch.zeros((max_steps, ac_dim)))
        #     self.rew_buffs.append(torch.zeros((max_steps, 1)))
        #     self.mc_buffs.append(torch.zeros((max_steps, 1)))
        #     self.next_obs_buffs.append(torch.zeros((max_steps, obs_dim)))
        #     self.done_buffs.append(torch.zeros((max_steps, 1)))
        #     self.n_step_buffs.append(torch.zeros((max_steps, 1)))
        #     self.ws_buffs.append(torch.zeros((max_steps, 1)))
            # self.SIL_priority.append(np.zeros(max_steps))

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def push():
        pass

    def sample():
        pass

    def merge_buffer():
        pass

    def __len__(self):
        return self.filled_i

    def getTotalDim(self, pretrain, obs_dim, ac_dim,k,hidden_dim_lstm,prox_item_size):
        if not pretrain:
            total_dim = obs_dim+ac_dim+6+k+(hidden_dim_lstm*4)+prox_item_size
        else:
            total_dim = obs_dim+ac_dim+6+k+prox_item_size

        return total_dim

    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]

    def get_cumulative_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].sum() for i in range(self.num_agents)]
    
    def update_priorities(self, agentID,inds, prio,k=1): #KEEP
        # self.ensemble_priorities[inds,agentID,k] = prio
        self.seq_exps[inds, :, agentID, self.obs_acs_dim+5+k] = prio
    
    def update_SIL_priorities(self, agentID,inds, prio):
        self.SIL_priorities[inds,agentID] = prio

    def get_PER_inds(self,agentID=0,batch_size=32,k=1):
        '''Returns a sample prioritized using TD error for the update and the indices used'''
        # inds = np.random.choice(np.arange(self.filled_i), size=N,
        #                         replace=False)   
        prios = self.ensemble_priorities[:self.filled_i,agentID,k].detach()
        if prios.sum() == 0.0:
            reset = np.ones(len(prios))/(1.0*len(prios))
            probs = reset/np.sum(reset)
        else:
            probs = prios.numpy()/(prios.sum().numpy())
            while np.abs(probs.sum() - 1) > 0.0003:
                probs = probs/probs.sum()
        return np.random.choice(self.filled_i,batch_size,p=probs,replace=False)
                              
    def get_SIL_inds(self,agentID=0,batch_size=32):
                              
        '''Returns a sample prioritized using MC_Targets - Q for the Self-Imitation update and the indices used'''
        # inds = np.random.choice(np.arange(self.filled_i), size=N,
        #                         replace=False)   
        prios = self.SIL_priorities[:self.filled_i,agentID,:].squeeze().detach()
        if prios.sum() == 0.0:
            reset = np.ones(len(prios))/(1.0*len(prios))
            probs = reset/np.sum(reset)
        else:
            probs = prios.numpy()/(prios.sum().numpy())
            while np.abs(probs.sum() - 1) > 0.0003:
                probs = probs/probs.sum()
        return np.random.choice(self.filled_i,batch_size,p=probs)
