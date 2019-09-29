
def initialize_buffer(_params: dict):
    return AbstractReplayBuffer()

class AbstractReplayBuffer():

    def __init__(self):
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

class BasicReplayBuffer(AbstractReplayBuffer):

    def __init__(self):
        pass

