import torch
import numpy as np
from torch.autograd import Variable

import collections

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

def initialize_buffer(config, num_agents, obs_space, act_space):
    return SimpleExperienceBuffer(config['max_size'], config['batch_size'])
    # return BasicReplayBuffer(config['max_size'], num_agents, obs_space, act_space)

class AbstractReplayBuffer(object):

    def __init__(self, max_size, num_agents, obs_dim, acs_dim):
        self.current_index = 0
        self.size = 0
        self.num_agents = num_agents
        self.max_size = max_size
        self.obs_buffer = torch.zeros((self.max_size, self.num_agents, obs_dim), requires_grad=False)
        self.ac_buffer = torch.zeros((self.max_size, self.num_agents, acs_dim),requires_grad=False)
        self.rew_buffer = torch.zeros((self.max_size, self.num_agents, 1),requires_grad=False)
        self.next_obs_buffer = torch.zeros((self.max_size, self.num_agents, obs_dim),requires_grad=False)
        self.done_buffer = torch.zeros((self.max_size, self.num_agents, 1),requires_grad=False)
        self.obs_dim = obs_dim
        self.acs_dim = acs_dim

    def __len__(self):
        return self.size

    def push(self):
        pass

    def sample(self):
        pass

    def clear(self):
        pass


##########################################################################
#
#    Simple Buffer for a Single Agents Experience
#
##########################################################################

class SimpleExperienceBuffer:
    def __init__(self, capacity, batch_size):
        self.buffer = collections.deque(maxlen=capacity)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def clear_buffer(self):
        self.buffer = collections.deque(maxlen=capacity)

    def sample(self):
        indices = np.random.choice(len(self.buffer), self.batch_size)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(next_states), np.array(dones, dtype=np.bool)

    def full_buffer(self):
        random_buffer = np.random.permutation(self.buffer)
        states, actions, rewards, next_states, dones = zip(*[random_buffer[idx] for idx in range(len(random_buffer))])
        return np.array(states),np.array(actions), np.array(rewards,dtype = np.float32), \
            np.array(next_states), np.array(dones, dtype=np.bool)


##########################################################################
#
#   Replay Buffer is capable of storing Multi-Agents
#   made by Daniel Tellier
#
##########################################################################

class MultiAgentReplayBuffer(AbstractReplayBuffer):

    def __init__(self, max_size, num_agents, obs_dim, acs_dim):
        super(MultiAgentReplayBuffer, self).__init__(max_size, num_agents, obs_dim, acs_dim)

    def push(self, exps):
        nentries = len(exps)

        if self.current_index + nentries > self.max_size:
            rollover = self.max_size - self.current_index
            self.obs_buffer = roll(self.obs_buffer, rollover)
            self.acs_buffer = roll(self.acs_buffer, rollover)
            self.rew_buffer = roll(self.rew_buffer, rollover)
            self.done_buffer = roll(self.done_buffer, rollover)
            self.next_obs_buffer = roll(self.next_obs_buffer, rollover)

            self.current_index = 0
            self.size = self.max_size

        action_i = self.obs_dim
        rew_i = action_i + self.acs_dim
        done_i = rew_i+1
        next_obs_i = done_i+1

        self.obs_buffer[self.current_index:self.current_index+nentries, :self.num_agents, :self.obs_dim] = exps[:, :, :self.obs_dim]
        self.ac_buffer[self.current_index:self.current_index+nentries, :self.num_agents, :self.ac_dim] = exps[:, :, action_i:rew_i]
        self.rew_buffer[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = exps[:, :, rew_i: done_i]
        self.done_buffer[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = exps[:, :, done_i:done_i+1]
        self.next_obs_buffer[self.current_index:self.current_index+nentries, :self.num_agents, :self.obs_dim] =  exps[:, :, next_obs_i:]

    def sample(self, inds, to_gpu=False, device='cpu'):
        if to_gpu:
            cast = lambda x: Variable(x, requires_grad=False).to(device)
            cast_obs = lambda x: Variable(x, requires_grad=True).to(device)
        else:
            cast = lambda x: Variable(x, requires_grad=False)
            cast_obs = lambda x: Variable(x, requires_grad=True)

        return (
                    [cast_obs(self.obs_buffs[inds, i, :]) for i in range(self.num_agents)],
                    [cast(self.ac_buffs[inds, i, :]) for i in range(self.num_agents)],
                    [cast(self.rew_buffs[inds, i, :]).squeeze() for i in range(self.num_agents)],
                    [cast(self.done_buffs[inds, i, :]).squeeze() for i in range(self.num_agents)],
                    [cast_obs(self.next_obs_buffs[inds, i, :]) for i in range(self.num_agents)]
                )
