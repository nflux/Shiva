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


class DQN_Learner(AbstractLearner):

    def __init__(self, 
                agents, 
                environments, 
                algorithm, 
                data
        ):
        self.agents = agents
        self.environments = environments
        self.algorithm = algorithm
        self.data = None


    def launch(self):
        # create the environment and get the action and observation spaces
        Environment.create_env(self.environments)
        Environment.get_obs_space()
        Environment.get_action_space()
        # somehow call the specific replay buffer that we want?
        buffer = Replay_Buffer.create_buffer()
        # Create an instance of the algorithm
        Algorithm.create_alg(self.algorithm)

    def step(self):
        observation = Environment.get_observation()
        action = Algorithm.get_action()
        Environment.step()
        Environment.get_reward()
        Environment.get_observation()

class MADDPG_Learner(AbstractLearner):
    
    def __init__(self):
        pass


class DDPG_Learner(AbstractLearner): 
        