import Algorithm
import Replay_Buffer
import Environment

class AbstractLearner():
    
    def __init__(self, agents, environments, algorithm, data):
        self.agents = agents
        self.environments = environments
        self.algorithm = algorithm
        self.data = None
    

    def update(self):
        self.algorithm.update(agents, data)

    def step(self):
        pass

    def create_env(self, alg):
        pass

    def get_obs_space(self):
        return Environment.get_obs_space()

    def get_agents(self):
        return self.agents

    def get_alg(self):
        return self.algorithm

    def launch(self):
        pass

    def save_agent(self):
        pass

    def load_agent(self):
        pass


class Learner(AbstractLearner):

    def __init__(self, agents, environments, algorithm, data):
        self.agents = agents
        self.environments = environments
        self.algorithm = algorithm
        self.data = None


    def launch(self):
        