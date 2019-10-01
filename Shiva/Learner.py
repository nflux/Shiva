import Algorithm 
from Replay_Buffer import initialize_buffer
import Environment

class AbstractLearner():

    def __init__(self, 
                agents : list, 
                environments : list, 
                algorithm : str, 
                data : list,
                configs: dict
                ):
        # Does all this even makes sense?
        # This objects are OVERWRITTEN in the launch()
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
        # Does all this makes sense?
        self.agents = agents
        self.environments = environments
        self.algorithm = algorithm
        self.data = None
        self.configs = configs

    def create_environment(self):
        # create the environment and get the action and observation spaces
        self.env = Environment.initialize_env(self.configs['Environment'])


    def get_agents(self):
        return self.agents

    def get_algorithm(self):
        return self.algorithm

    def launch(self):
        
        # Launch the environment
        self.create_environment()
        
        # Launch the algorithm which will handle the 
        self.alg = Algorithm.initialize_algorithm(self.env.get_observation_space(), self.env.get_action_space(), [self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])
        self.agents = self.alg.get_agents()
        # Basic replay buffer at the moment
        self.buffer = initialize_buffer(self.configs['Replay_Buffer'], len(self.agents), self.env.get_action_space(), self.env.get_observation_space())



        print('Launch done.')

    def step(self):

        # probably need to discuss this with Ezequiel
        observation = self.env.get_observation(0)

        action = self.alg.get_action(self.agents[0], observation, self.env.get_current_step())

        next_observation, reward, done = self.env.step(action)

        self.buffer.push([[[*observation, action, reward, *next_observation, int(done)]]])

        print('Step')
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