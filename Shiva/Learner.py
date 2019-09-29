import Algorithm 
import Replay_Buffer
import Environment 

class AbstractLearner():

    def __init__(self, 
                agents : list, 
                environments : list, 
                algorithm : str, 
                data : list,
                configs: dict
                ):

        self.agents = agents
        self.environments = environments
        self.algorithm = algorithm
        self.data = None


    '''
        not sure if the algorithm should play that big of a role here

        if self.algorithm == 'DQN':
            learner = DQN_Learner(agents, environments, algorithm, data)

        elif self.algorithm == 'A3C':
            leaner = A3C_Learner(agents, environments, algorithm, data)

        elif self.algorithm == 'DDPG':
            leaner = DDPG_Learner(agents, environments, algorithm, data)

        elif self.algorithm == 'MADDPG':
            leaner = MADDPG_Learner(agents, environments, algorithm, data)
    '''

    def update(self):
        pass

    def step(self):
        pass

    def create_env(self, alg):
        pass

    def get_agents(self):
        pass

    def get_algorithm(self):
        pass

    def launch(self):
        pass

    def save_agent(self):
        pass

    def load_agent(self):
        pass


class Learner(AbstractLearner):

    def __init__(self, agents, environments, algorithm, data, configs):
        self.agents = agents
        self.environments = environments
        self.algorithm = algorithm
        self.data = None
        self.configs = configs
        print(self.configs['Algorithm'])

    def create_environment(self, alg):
        # create the environment and get the action and observation spaces
        return Environment(self.configs['Environment'],len(self.agents))


    def get_agents(self):
        return self.agents

    def get_algorithm(self):
        return self.algorithm

    def launch(self):

        env = self.create_environment()
        env.get_obs_space()
        env.get_action_space()

        self.create_environment(self.algorithm)
        # somehow call the specific replay buffer that we want?
        buffer = Replay_Buffer.create_buffer()
        # Create an instance of the algorithm
        Algorithm.create_alg(self.algorithm)

    def step(self):

        # probably need to discuss this with Ezequiel
        observation = Environment.get_observation()
        action = Algorithm.get_action()
        Environment.step(action)
        reward = Environment.get_reward()
        next_observation = Environment.get_observation()
        Replay_Buffer.push(observation)

    def save_agent(self):
        pass

    def load_agent(self):
        pass

'''

Maybe the learner shouldn't be tied to an algorithm

class DDPG_Learner(AbstractLearner):
    
    def __init__(self):
        pass


class MADDPG_Learner(AbstractLearner):
    
    def __init__(self):
        pass


class A3C_Learner(AbstractLearner):
    
    def __init__(self):
        pass

'''