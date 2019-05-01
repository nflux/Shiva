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

def init_buffer(config, lstm_flag, obs_dim,prox_item_size,pretrain=False):
    buffer = None
    if lstm_flag:
        buffer = ReplayBufferLSTM(config,obs_dim,prox_item_size,pretrain)
    else:
        buffer = ReplayBufferNonLSTM(config,obs_dim,prox_item_size,pretrain)

    return buffer

class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, config,obs_dim,prox_item_size,pretrain=False):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.config = config
        self.max_steps = config.replay_size
        # self.max_episodes = int(max_steps/episode_length)
        self.num_agents = config.num_left

        self.obs_dim = obs_dim
        self.ac_dim = config.ac_dim
        self.obs_acs_dim = obs_dim + self.ac_dim 
        self.hidden_dim_lstm = config.hidden_dim_lstm 
        self.prox_item_size = prox_item_size   
        self.pretrain = pretrain
        self.k = config.k_ensembles
        self.SIL = config.sil
        self.total_dim = self.getTotalDim(self.pretrain)
        
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

    def getTotalDim(self, pretrain):
        if not pretrain:
            total_dim = self.obs_dim+self.ac_dim+6+self.k+(self.hidden_dim_lstm*4)+self.prox_item_size
        else:
            total_dim = self.obs_dim+self.ac_dim+6+self.k+self.prox_item_size

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

class ReplayBufferNonLSTM(ReplayBuffer):
    """
    Non-LSTM Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, config, obs_dim,prox_item_size,pretrain=False):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        #This initializes the parent buffer methods/variables
        super().__init__(config, obs_dim,prox_item_size,pretrain)
        
        # self.max_episodes = int(max_steps/episode_length)
        self.start_loc = 0
        self.obs_acs_dim = obs_dim + self.ac_dim
        self.prox_item_size_per_agent = prox_item_size//(2*self.num_agents)

        self.obs_buffs = torch.zeros((self.max_steps, self.num_agents, obs_dim))
        self.ac_buffs = torch.zeros((self.max_steps, self.num_agents, self.ac_dim),requires_grad=False)
        self.n_step_buffs = torch.zeros((self.max_steps, self.num_agents, 1),requires_grad=False)
        self.rew_buffs = torch.zeros((self.max_steps, self.num_agents, 1),requires_grad=False)
        self.mc_buffs = torch.zeros((self.max_steps, self.num_agents, 1),requires_grad=False)
        self.next_obs_buffs = torch.zeros((self.max_steps, self.num_agents, obs_dim),requires_grad=False)
        self.done_buffs = torch.zeros((self.max_steps, self.num_agents, 1),requires_grad=False)
        self.ws_buffs = torch.zeros((self.max_steps, self.num_agents, 1),requires_grad=False)
        self.SIL_priorities = torch.zeros((self.max_steps,self.num_agents,1),requires_grad=False)
        self.ensemble_priorities = torch.zeros((self.max_steps,self.num_agents,self.k),requires_grad=False)
    
    def push(self, exps):
        nentries = len(exps)
    
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            self.obs_buffs = roll(self.obs_buffs, rollover)
            self.ac_buffs = roll(self.ac_buffs, rollover)
            self.rew_buffs = roll(self.rew_buffs, rollover)
            self.next_obs_buffs = roll(self.next_obs_buffs, rollover)
            self.done_buffs = roll(self.done_buffs, rollover)
            self.mc_buffs = roll(self.mc_buffs, rollover)
            self.n_step_buffs = roll(self.n_step_buffs, rollover)
            self.ws_buffs = roll(self.ws_buffs, rollover)
            self.ensemble_priorities = roll(self.ensemble_priorities,rollover)
            self.SIL_priorities = roll(self.SIL_priorities,rollover)


            self.curr_i = 0
            self.filled_i = self.max_steps

        # Used just for readability
        oa_i = self.obs_dim + self.ac_dim
        rew_i = oa_i+1
        next_oi = rew_i+self.obs_dim

        self.obs_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :self.obs_dim] = exps[:, :, :self.obs_dim]
        self.ac_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :self.ac_dim] = exps[:, :, self.obs_dim:oa_i]
        self.rew_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = exps[:, :, oa_i:rew_i]
        self.next_obs_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :self.obs_dim] = exps[:, :, rew_i:next_oi]
        self.done_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = exps[:, :, next_oi:next_oi+1]
        self.mc_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = exps[:, :, next_oi+1:next_oi+2]
        self.n_step_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = exps[:, :, next_oi+2:next_oi+3]
        self.ws_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = exps[:, :, next_oi+3:next_oi+4]
        self.ensemble_priorities[self.curr_i:self.curr_i+nentries, :self.num_agents, :self.k] = exps[:, :, next_oi+4:next_oi+4+self.k]
        self.SIL_priorities[self.curr_i:self.curr_i+nentries, :self.num_agents, :self.k] = exps[:, :, next_oi+4+self.k:next_oi+4+self.k+1]

        
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0
       
    
    def merge_buffer(self, buffer):
        nentries = len(buffer)
    
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            self.obs_buffs = roll(self.obs_buffs, rollover)
            self.ac_buffs = roll(self.ac_buffs, rollover)
            self.rew_buffs = roll(self.rew_buffs, rollover)
            self.next_obs_buffs = roll(self.next_obs_buffs, rollover)
            self.done_buffs = roll(self.done_buffs, rollover)
            self.mc_buffs = roll(self.mc_buffs, rollover)
            self.n_step_buffs = roll(self.n_step_buffs, rollover)
            self.ws_buffs = roll(self.ws_buffs, rollover)
            self.ensemble_priorities = roll(self.ensemble_priorities,rollover)
            self.SIL_priorities = roll(self.SIL_priorities,rollover)


            self.curr_i = 0
            self.filled_i = self.max_steps

        # Used just for readability
        oa_i = self.obs_dim + self.ac_dim
        rew_i = oa_i+1
        next_oi = rew_i+self.obs_dim

        self.obs_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :self.obs_dim] = buffer.obs_buffs[:nentries, :self.num_agents, :self.obs_dim]
        self.ac_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :self.ac_dim] = buffer.ac_buffs[:nentries, :self.num_agents, :self.ac_dim]
        self.rew_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = buffer.rew_buffs[:nentries, :self.num_agents, :1]
        self.next_obs_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :self.obs_dim] = buffer.next_obs_buffs[:nentries, :self.num_agents, :self.obs_dim]
        self.done_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = buffer.done_buffs[:nentries, :self.num_agents, :1]
        self.mc_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = buffer.mc_buffs[:nentries, :self.num_agents, :1]
        self.n_step_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] =buffer.n_step_buffs[:nentries, :self.num_agents, :1]
        self.ws_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = buffer.ws_buffs[:nentries, :self.num_agents, :1]
        self.ensemble_priorities[self.curr_i:self.curr_i+nentries, :self.num_agents, :self.k] = buffer.ensemble_priorities[:nentries, :self.num_agents, : self.k]
        self.SIL_priorities[self.curr_i:self.curr_i+nentries, :self.num_agents, :self.k] = buffer.SIL_priorities[:nentries, :self.num_agents,  :self.k]

        
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self,inds, to_gpu=False, norm_rews=False,device="cuda:0"):

        if to_gpu:
            cast = lambda x: Variable(x,requires_grad=False).to(device)
        else:
            cast = lambda x: Variable(x, requires_grad=False)
        if to_gpu:
            cast_obs = lambda x: Variable(x,requires_grad=True).to(device) # obs need gradient for cent-Q
        else:
            cast_obs = lambda x: Variable(x, requires_grad=True)


        if norm_rews:
            ret_rews = [cast((self.rew_buffs[inds, i, :] -
                             self.rew_buffs[:self.filled_i, i, :].mean()) /
                             self.rew_buffs[:self.filled_i, i, :].std())
                        for i in range(self.num_agents)]

            ret_mc = [cast((self.mc_buffs[inds, i, :] -
                              self.mc_buffs[:self.filled_i, i, :].mean()) /
                             self.mc_buffs[:self.filled_i, i, :].std())
                        for i in range(self.num_agents)]

            ret_n_step = [cast((self.n_step_buffs[inds, i, :] -
                              self.n_step_buffs[:self.filled_i, i, :].mean()) /
                             self.n_step_buffs[:self.filled_i, i, :].std())
                        for i in range(self.num_agents)]
            
        else:
            ret_rews = [cast(self.rew_buffs[inds, i, :]).squeeze() for i in range(self.num_agents)]
            ret_mc = [cast(self.mc_buffs[inds, i, :]).squeeze() for i in range(self.num_agents)]
            ret_n_step = [cast(self.n_step_buffs[inds, i, :]).squeeze() for i in range(self.num_agents)]


        return ([cast_obs(self.obs_buffs[inds, i, :]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[inds, i, :]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[inds, i, :]) for i in range(self.num_agents)],
                [cast(self.done_buffs[inds, i, :]).squeeze() for i in range(self.num_agents)],
                ret_mc,
                ret_n_step,
                [cast(self.ws_buffs[inds, i, :]).squeeze() for i in range(self.num_agents)])

class ReplayBufferLSTM(ReplayBuffer):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, config, obs_dim,prox_item_size,pretrain=False):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        #This initializes the parent buffer methods/variables
        super().__init__(config,obs_dim,prox_item_size,pretrain)
        
        self.seq_length = config.seq_length
        self.overlap = config.overlap
        
        self.max_steps = int(config.replay_size) # Max sequences did this to reduce var name change
        self.seq_exps = torch.zeros((self.seq_length, self.max_steps, self.num_agents, self.total_dim), requires_grad=False)

    def merge_buffer(self, buffer):
        nentries = len(buffer)

        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            self.seq_exps = roll2(self.seq_exps, rollover)

            self.curr_i = 0
            self.filled_i = self.max_steps
        
        self.seq_exps[:self.seq_length, self.curr_i:self.curr_i+nentries, :self.num_agents, :] = buffer.seq_exps[:self.seq_length, :nentries, :self.num_agents, :]

        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def push(self, exps):
        ep_length = len(exps)
        nentries = (math.ceil(ep_length/self.overlap)) - 1 # Number of rows to input in seq_exps tensor
    
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            self.seq_exps = roll2(self.seq_exps, rollover)

            self.curr_i = 0
            self.filled_i = self.max_steps

        # NOTE: Exps include: obs, acs, n_step_rew, n_step_done, MC_targets, n_step_targets, 
        #       n_step_ws, ensemble priorities, replay priorities
        if ep_length % self.overlap == 0:
            for n in range(nentries):
                start_pt = n*self.overlap
                self.seq_exps[:self.seq_length, self.curr_i+n, :self.num_agents, :] = exps[start_pt:start_pt+self.seq_length, :, :]
        else:
            for n in range(nentries):
                if n != nentries-1:
                    start_pt = n*self.overlap
                    self.seq_exps[:self.seq_length, self.curr_i+n, :self.num_agents, :] = exps[start_pt:start_pt+self.seq_length, :, :]
                else:
                    # Get the last values if the episode length is not evenly divisible by the overlap amount
                    self.seq_exps[:self.seq_length, self.curr_i+n, :self.num_agents, :] = exps[ep_length-self.seq_length:, :, :]

        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, inds, to_gpu=False, device="cuda:0"):

        if to_gpu:
            cast = lambda x: Variable(x, requires_grad=False).to(device)
        else:
            cast = lambda x: Variable(x, requires_grad=False)
        if to_gpu:
            cast_obs = lambda x: Variable(x, requires_grad=True).to(device) # obs need gradient for cent-Q
        else:
            cast_obs = lambda x: Variable(x, requires_grad=True)

        if not self.pretrain:
            prox_start = self.obs_acs_dim+5+1+self.k+self.hidden_dim_lstm*4
        else:
            prox_start = self.obs_acs_dim +6 + self.k
        prox_item_list = [0, self.obs_dim*self.num_agents, self.obs_dim*2*self.num_agents, self.ac_dim + self.obs_dim*2*self.num_agents]
        return ([cast_obs(self.seq_exps[:, inds, a, :self.obs_dim]) for a in range(self.num_agents)], # obs
                [cast(self.seq_exps[:, inds, a, self.obs_dim:self.obs_acs_dim]) for a in range(self.num_agents)], # actions
                [cast(self.seq_exps[:, inds, a, self.obs_acs_dim:self.obs_acs_dim+1]) for a in range(self.num_agents)], # rewards
                [cast(self.seq_exps[:, inds, a, self.obs_acs_dim+1:self.obs_acs_dim+2]) for a in range(self.num_agents)], # dones
                [cast(self.seq_exps[:, inds, a, self.obs_acs_dim+2:self.obs_acs_dim+3]) for a in range(self.num_agents)], # mc
                [cast(self.seq_exps[:, inds, a, self.obs_acs_dim+3:self.obs_acs_dim+4]) for a in range(self.num_agents)], # n_step_targets
                [cast(self.seq_exps[:, inds, a, self.obs_acs_dim+4:self.obs_acs_dim+5]) for a in range(self.num_agents)], # ws
                self.seq_exps[0, inds, 0, self.obs_acs_dim+6+self.k:self.obs_acs_dim+6+self.k+self.hidden_dim_lstm*4], # recurrent states for both critics
                [[[cast_obs(self.seq_exps[:, inds, outer, prox_start+(inner*self.obs_dim)+prox_item_list[p]:prox_start+(inner*self.obs_dim)+prox_item_list[p]+self.obs_dim])
                if p <= 1 else cast(self.seq_exps[:, inds, outer, prox_start+(inner*self.ac_dim)+prox_item_list[p]:prox_start+(inner*self.ac_dim)+prox_item_list[p]+self.ac_dim])
                for inner in range(self.num_agents)] for p in range(len(prox_item_list))] for outer in range(self.num_agents)])