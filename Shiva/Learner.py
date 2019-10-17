import Algorithm
import Replay_Buffer
import Environment

from tensorboardX import SummaryWriter

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


    def update(self):

        print("Before training")
        
        for ep_count in range(self.configs['Learner']['episodes']):
            self.env.reset()
            self.totalReward = 0
            done = False
            while not done:
                done = self.step(ep_count)

        self.env.env.close()

        print("After training")


    def step(self, ep_count):

        self.env.env.render()

        observation = self.env.get_observation()

        action = self.alg.get_action(self.alg.agents[0], observation, self.env.get_current_step())

        next_observation, reward, done = self.env.step(action)

        self.writer.add_scalar('Reward', reward, self.env.get_current_step())

        self.totalReward += reward

        self.buffer.append([observation, action, reward, next_observation, done])

        self.alg.update(self.agents, self.buffer.sample(), self.env.get_current_step())

        if done:
            self.writer.add_scalar('Total Reward', self.totalReward, ep_count)
            self.writer.add_scalar('Average Actor Loss per Episode', self.alg.get_average_actor_loss(self.env.get_current_step()), ep_count)
            self.writer.add_scalar('Average Critic Loss per Episode', self.alg.get_average_critic_loss(self.env.get_current_step()), ep_count)

        return done

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

        self.writer = SummaryWriter()

        # Basic replay buffer at the moment
        self.buffer = Replay_Buffer.initialize_buffer(self.configs['Replay_Buffer'], 1, self.env.get_action_space(), self.env.get_observation_space())

        print('Launch done.')


    def save_agent(self):
        pass

    def load_agent(self):
        pass




class Single_Agent_Imitation_Learner(AbstractLearner):
    def __init__(self):

        super(ImitationLearner,self).__init(agents,environments,algorithm,data,configs)
        self.agents = agents
        self.environments = environments
        self.algorithm = algorithm
        self.data = data
        self.configs = configs

    def create_environment(self):
        #Initialize environment-config file will specify which environment type
        self.env = Environment.initialize_env(self.configs['Environment'])

    def get_agents(self):
        return self.agents[0]

    def get_algorithm(self):
        return self.Algorithm

    def launch(self):
        pass