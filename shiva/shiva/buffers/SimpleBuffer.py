

class SimpleBuffer:
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