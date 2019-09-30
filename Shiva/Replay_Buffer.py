import torch
import numpy as np

def initialize_buffer(_params: dict):
    return AbstractReplayBuffer()

class AbstractReplayBuffer():

    def __init__(self,
                max_size : int,
                num_agents : int,
                obs_dim : int,
                acs_dim : int,
                current_index : int,
                filled_index : int,
                obs_storage : torch.Tensor(),
                acs_storage : torch.Tensor(),
                rew_storage : torch.Tensor(),
                done_storage : torch.Tensor()
                ):
        pass

    def push(self):
        pass

    def sample(self, size, aux: set() ):
        # aux is of the form (n_obs, observation_space, action_space)
        n_obs = aux[0]
        observation_space = aux[1]
        action_space = aux[2]
        
        states = np.array(torch.rand(n_obs, observation_space))
        actions = np.array(torch.randint(0, action_space-1, (n_obs,)))
        rewards = np.array(torch.rand(n_obs))
        done = np.array(torch.randint(0, 2, (n_obs,)))
        next_state = np.array(torch.rand(n_obs, observation_space))
        return [states, actions, rewards, done, next_state]

    def roll(self):
        pass
    
    def clear(self):
        pass

class BasicReplayBuffer(AbstractReplayBuffer):

    def __init__(self):
        pass

