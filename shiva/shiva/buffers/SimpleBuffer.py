import numpy as np
import collections
# from buffers import ReplayBuffer

class SimpleBuffer():

    '''
        Expected Configs:

            batch_size
            capacity    

    '''

    def __init__(self, batch_size, capacity):
        # super(SimpleBuffer, self).__init__(max_size, num_agents, obs_dim, acs_dim)
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=self.capacity)
        self.batch_size = batch_size


    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def clear_buffer(self):
        self.buffer = collections.deque(maxlen=self.capacity)

    def sample(self):
        indices = np.random.choice(len(self.buffer), self.batch_size)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(next_states), np.array(dones, dtype=np.bool)

class SimpleRoboCupBuffer():

    '''
        Expected Configs:

            batch_size
            capacity    

    '''

    def __init__(self, batch_size, capacity):
        # super(SimpleBuffer, self).__init__(max_size, num_agents, obs_dim, acs_dim)
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=self.capacity)
        self.batch_size = batch_size


    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def clear_buffer(self):
        self.buffer = collections.deque(maxlen=self.capacity)

    def sample(self):
        indices = np.random.choice(len(self.buffer), self.batch_size)
        states, actions, rewards, next_states, dones, bot_actions = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(next_states), np.array(dones, dtype=np.bool), np.array(bot_actions)