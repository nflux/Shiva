class Agent:
    def __init__(self, obs_dim, action_dim, uid):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.id = uid # unique id
        self.policy = None
        self.target_policy = None
    
    def policy(self, obs):
        pass

    def save(self):
        pass

    def load(self):
        pass