import numpy as np
from torch import Tensor,squeeze
from torch.autograd import Variable
import operator
import torch

def roll(tensor, rollover):
    '''
    Roll over the first axis of a tensor
    '''
    return torch.cat((tensor[-rollover:], tensor[:-rollover]))


class ReplayTensorBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, num_agents, obs_dim, ac_dim, batch_size, LSTM, LSTM_PC):
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
        self.start_loc = 0
        self.obs_dim = obs_dim
        self.ac_dim = ac_dim
        self.obs_buffs = torch.zeros((max_steps, num_agents, obs_dim))
        self.ac_buffs = torch.zeros((max_steps, num_agents, ac_dim))
        self.n_step_buffs = torch.zeros((max_steps, num_agents, 1))
        self.rew_buffs = torch.zeros((max_steps, num_agents, 1))
        self.mc_buffs = torch.zeros((max_steps, num_agents, 1))
        self.next_obs_buffs = torch.zeros((max_steps, num_agents, obs_dim))
        self.done_buffs = torch.zeros((max_steps, num_agents, 1))
        self.ws_buffs = torch.zeros((max_steps, num_agents, 1))
        # self.SIL_priority = []
        self.episode_buff = []
        self.done_step = False
        self.batch_size =  batch_size
        self.count = 0
        self.LSTM = LSTM
        self.LSTM_PC = LSTM_PC
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

    def __len__(self):
        return self.filled_i
    
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
            # self.SIL_priority[agent_i] = np.roll(self.SIL_priority[agent_i],
            #                                          rollover)

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
        
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0
        

    def push_LSTM(self, observations, actions, rewards, next_observations, dones,mc_targets,n_step,ws):
        #nentries = observations.shape[0]  # handle multiple parallel environments
        # for now ** change **
        nentries = 1
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i],
                                                  rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i],
                                                 rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i],
                                                  rollover)
                self.mc_buffs[agent_i] = np.roll(self.mc_buffs[agent_i],
                                                  rollover)
                self.next_obs_buffs[agent_i] = np.roll(
                    self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i],
                                                   rollover)
                self.n_step_buffs[agent_i] = np.roll(self.n_step_buffs[agent_i],
                                                 rollover)
                self.ws_buffs[agent_i] = np.roll(self.ws_buffs[agent_i],
                                                   rollover)
                self.SIL_priority[agent_i] = np.roll(self.SIL_priority[agent_i],
                                                     rollover)

            self.curr_i = 0
            self.filled_i = self.max_steps
        for agent_i in range(self.num_agents):

            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                observations[:, agent_i]).T # added .T
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[agent_i]
            self.mc_buffs[agent_i][self.curr_i:self.curr_i + nentries] = mc_targets[agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                next_observations[:, agent_i]).T # added .T
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones[agent_i]
            self.n_step_buffs[agent_i][self.curr_i:self.curr_i + nentries] = n_step[agent_i]
            self.ws_buffs[agent_i][self.curr_i:self.curr_i + nentries] = ws[agent_i]
            self.SIL_priority[agent_i][self.curr_i:self.curr_i +nentries] = 1.0
        
        # Track experience based off the episodes
        # print('This is the dones',str(dones[0]))
        if self.done_step == True:
            if len(self.episode_buff) >= self.max_episodes:
                self.episode_buff[0:(1+len(self.episode_buff))-self.max_episodes] = []
            self.episode_buff.append(
                {
                    'obs': [self.obs_buffs[a][self.start_loc:self.curr_i+1] for a in range(self.num_agents)],
                    'acs': [self.ac_buffs[a][self.start_loc:self.curr_i+1] for a in range(self.num_agents)],
                    'rew': [self.rew_buffs[a][self.start_loc:self.curr_i+1] for a in range(self.num_agents)],
                    'mc': [self.mc_buffs[a][self.start_loc:self.curr_i+1] for a in range(self.num_agents)],
                    'next_obs': [self.next_obs_buffs[a][self.start_loc:self.curr_i+1] for a in range(self.num_agents)],
                    'dones': [self.done_buffs[a][self.start_loc:self.curr_i+1] for a in range(self.num_agents)],
                    'n_step': [self.n_step_buffs[a][self.start_loc:self.curr_i+1] for a in range(self.num_agents)],
                    'ws': [self.ws_buffs[a][self.start_loc:self.curr_i+1] for a in range(self.num_agents)],
                    'ep_length': self.curr_i - self.start_loc + 1
                }
            )

            self.start_loc = self.curr_i+1
            self.done_step = False


        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            roll_amount = self.curr_i - (self.start_loc+1)
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i],
                                                  roll_amount, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i],
                                                 roll_amount, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i],
                                                  roll_amount)
                self.mc_buffs[agent_i] = np.roll(self.mc_buffs[agent_i],
                                                  roll_amount)
                self.next_obs_buffs[agent_i] = np.roll(
                    self.next_obs_buffs[agent_i], roll_amount, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i],
                                                   roll_amount)
                self.n_step_buffs[agent_i] = np.roll(self.n_step_buffs[agent_i],
                                                 roll_amount)
                self.ws_buffs[agent_i] = np.roll(self.ws_buffs[agent_i],
                                                   roll_amount)
                self.SIL_priority[agent_i] = np.roll(self.SIL_priority[agent_i],
                                                     roll_amount)
            self.curr_i = roll_amount
            self.start_loc = 0

    def sample(self,inds, to_gpu=False, norm_rews=False,):

        if to_gpu:
            cast = lambda x: Variable(x,requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(x, requires_grad=False)
        if to_gpu:
            cast_obs = lambda x: Variable(x,requires_grad=True).cuda() # obs need gradient for cent-Q
        else:
            cast_obs = lambda x: Variable(x, requires_grad=True)


        if norm_rews:
            ret_rews = [cast((self.rew_buffs[inds, i, :] -
                             self.rew_buffs[:self.filled_i, i, :].mean()) /
                             self.rew_buffs[:self.filled_i, i, :].std())
                        for i in range(self.num_agents)]

            ret_mc = [cast((self.mc_buffs[inds, i:i+1, :] -
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
    
    def sample_LSTM(self, inds, trace_length, to_gpu=False, norm_rews=False):
        # inds = np.random.choice(np.arange(self.filled_i), size=N,
        #                         replace=False)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if to_gpu:
            cast_obs = lambda x: Variable(Tensor(x), requires_grad=True).cuda() # obs need gradient for cent-Q
        else:
            cast_obs = lambda x: Variable(Tensor(x), requires_grad=True)

        points = [np.random.randint(0,self.episode_buff[i]['ep_length']+1-trace_length) for i in inds]

        if norm_rews:
            ret_rews = [cast([(self.episode_buff[i]['rew'][a][p:p+1] -
                self.episode_buff[i]['rew'][a][:self.filled_i].mean()) /
                self.episode_buff[i]['rew'][a][:self.filled_i].std()
                for i,p in zip(inds, points)]) for a in range(self.num_agents)]
            
            ret_mc = [cast([(self.episode_buff[i]['mc'][a][p:p+1] - 
                self.episode_buff[i]['mc'][a][:self.filled_i].mean()) /
                self.episode_buff[i]['mc'][a][:self.filled_i].std()
                for i,p in zip(inds, points)]) for a in range(self.num_agents)]
            
            ret_n_step = [cast([(self.episode_buff[i]['n_step'][a][p:p+1] - 
                self.episode_buff[i]['n_step'][a][:self.filled_i].mean()) /
                self.episode_buff[i]['n_step'][a][:self.filled_i].std()
                for i,p in zip(inds, points)]) for a in range(self.num_agents)]
        else:
            ret_rews = [cast([self.episode_buff[i]['rew'][a][p:p+1]
                for i,p in zip(inds,points)]) for a in range(self.num_agents)]
            ret_mc = [cast([self.episode_buff[i]['mc'][a][p:p+1]
                for i,p in zip(inds,points)]) for a in range(self.num_agents)]
            ret_n_step = [cast([self.episode_buff[i]['n_step'][a][p:p+1]
                for i,p in zip(inds,points)]) for a in range(self.num_agents)]

        ret_obs = [cast_obs([self.episode_buff[i]['obs'][a][p:p+trace_length]
            for i,p in zip(inds,points)]) for a in range(self.num_agents)]
        
        if self.LSTM_PC:
            ret_acs = [cast([self.episode_buff[i]['acs'][a][p:p+trace_length]
                for i,p in zip(inds,points)]) for a in range(self.num_agents)]
        else:
            ret_acs = [cast([self.episode_buff[i]['acs'][a][p:p+1]
                for i,p in zip(inds,points)]) for a in range(self.num_agents)]
        ret_next_obs = [cast_obs([self.episode_buff[i]['next_obs'][a][p:p+trace_length]
            for i,p in zip(inds,points)]) for a in range(self.num_agents)]
        ret_dones = [cast([self.episode_buff[i]['dones'][a][p:p+1]
            for i,p in zip(inds,points)]) for a in range(self.num_agents)]
        ret_ws = [cast([self.episode_buff[i]['ws'][a][p:p+1]
            for i,p in zip(inds,points)]) for a in range(self.num_agents)]

        if self.LSTM_PC:
            return ([ret_obs[a].permute(1,0,2) for a in range(self.num_agents)],
                    [ret_acs[a].permute(1,0,2) for a in range(self.num_agents)],
                    [ret_rews[a].view(self.batch_size) for a in range(self.num_agents)],
                    [ret_next_obs[a].permute(1,0,2) for a in range(self.num_agents)],
                    [ret_dones[a].view(self.batch_size) for a in range(self.num_agents)],
                    [ret_mc[a].view(self.batch_size) for a in range(self.num_agents)],
                    [ret_n_step[a].view(self.batch_size) for a in range(self.num_agents)],
                    [ret_ws[a].view(self.batch_size) for a in range(self.num_agents)])
        else:
            return ([ret_obs[a].permute(1,0,2) for a in range(self.num_agents)],
                    [ret_acs[a].view((self.batch_size, -1)) for a in range(self.num_agents)],
                    [ret_rews[a].view(self.batch_size) for a in range(self.num_agents)],
                    [ret_next_obs[a].permute(1,0,2) for a in range(self.num_agents)],
                    [ret_dones[a].view(self.batch_size) for a in range(self.num_agents)],
                    [ret_mc[a].view(self.batch_size) for a in range(self.num_agents)],
                    [ret_n_step[a].view(self.batch_size) for a in range(self.num_agents)],
                    [ret_ws[a].view(self.batch_size) for a in range(self.num_agents)])

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

    


    def sample_SIL(self,agentID=0,batch_size=32,to_gpu=False, norm_rews=False):
        '''Returns a sample prioritized using MC_Targets - Q for the Self-Imitation update and the indices used'''
        # inds = np.random.choice(np.arange(self.filled_i), size=N,
        #                         replace=False)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if to_gpu:
            cast_obs = lambda x: Variable(Tensor(x), requires_grad=True).cuda() # obs need gradient for cent-Q
        else:
            cast_obs = lambda x: Variable(Tensor(x), requires_grad=True)
            
            
        prios = self.SIL_priority[agentID][:self.filled_i]
        if prios.sum() == 0:
            probs = [1.0/len(prios)]*len(prios)
        else:
            probs = prios/prios.sum()

        inds  = np.random.choice(self.filled_i,batch_size,p=probs)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)]
        if norm_rews:
            ret_mc = [cast((self.mc_buffs[i][inds] -
                              self.mc_buffs[i][:self.filled_i].mean()) /
                             self.mc_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)]
        if norm_rews:
            ret_n_step = [cast((self.n_step_buffs[i][inds] -
                              self.n_step_buffs[i][:self.filled_i].mean()) /
                             self.n_step_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)]

        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
            ret_mc = [cast(self.mc_buffs[i][inds]) for i in range(self.num_agents)]
            ret_n_step = [cast(self.n_step_buffs[i][inds]) for i in range(self.num_agents)]


       
        return ([cast_obs(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)],
                ret_mc,
                ret_n_step,
                [cast(self.ws_buffs[i][inds]) for i in range(self.num_agents)]),inds
    
    def update_priorities(self, agentID,inds, prio):
        for i, p in zip(inds,prio):
            self.SIL_priority[agentID][i] = p

    