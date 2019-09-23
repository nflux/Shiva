class Learner():
    
    def __init__(self, agents, environments, algorithm, data):
        self.agents = agents
        self.environments = environments
        self.algorithm = algorithm
        self.data = data
    

    def update(self):
        self.algorithm.update(agents, data)

    def step(self):
        pass

    def create_env(self, alg):
        pass

    def get_agents():
        pass

    def get_alg():
        return self.algorithm

    def launch():
        pass

    def save_agent():
        pass

    def load_agent():
        pass