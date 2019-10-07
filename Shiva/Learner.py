import Algorithm 
import Replay_Buffer
import Environment

from abc import ABC

class AbstractLearner(ABC):

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


class Single_Agent_Learner(AbstractLearner):

    

    def __init__(self, agents, environments, algorithm, data, configs):

        super(Single_Agent_Learner,self).__init__(agents, environments, algorithm, data, configs)
        self.agents = agents
        self.environments = environments
        self.algorithm = algorithm
        self.data = None
        self.configs = configs

    def create_environment(self):
        # create the environment and get the action and observation spaces
        self.env = Environment.initialize_env(self.configs['Environment'])


    def get_agents(self):
        return self.agents[0]

    def get_algorithm(self):
        return self.algorithm

    def launch(self):
        
        # Launch the environment
        self.create_environment()
        
        # Launch the algorithm which will handle the 
        self.alg = Algorithm.initialize_algorithm(self.env.get_observation_space(), self.env.get_action_space(), [self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])
        
        self.agents = self.alg.create_agent()
        
        # Basic replay buffer at the moment
        self.buffer = Replay_Buffer.initialize_buffer(self.configs['Replay_Buffer'], 1, self.env.get_action_space(), self.env.get_observation_space())

        print('Launch done.')
        
    def step(self):

        self.env.env.render()
        
        observation = self.env.get_observation()

        action = self.alg.get_action(self.alg.agents[0], observation, self.env.get_current_step())

        next_observation, reward, done = self.env.step(action)

        self.buffer.append([observation, action, reward, next_observation, done])

        self.alg.update(self.agents, self.buffer.sample(), self.env.get_current_step())

        return done

    def update(self):


        print("Before training")
        for _ in range(self.configs['Learner']['episodes']):
            done = False
            self.env.reset()
            while not done:
                done = self.step()

        self.env.env.close()

        print("After training")
        

        pass

    def save_agent(self):
        pass

    def load_agent(self):
        pass
