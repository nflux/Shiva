from settings import shiva
import Algorithm
import ReplayBuffer
import Environment

def initialize_learner(config):
    if config[0]['Learner']['type'] == 'DQN':
        return SingleAgentMetaLearner([],
                                    config[0]['Algorithm'], # 
                                    config[0]['MetaLearner']['eval_env'],  # the evaluation environment
                                    [],  # this would have to be a list of agent objects
                                    [],  # this would have to be a list of elite agents objects
                                    config[0]['MetaLearner']['optimize_env_hp'],
                                    config[0]['MetaLearner']['optimize_learner_hp'],
                                    config[0]['MetaLearner']['evolution'],
                                    config,
                                    )

class AbstractLearner():
    
    def __init__(self,
                id: "An id generated by metalearner to index the learner.",
                agents : list,
                environments : list,
                algorithm : str,
                data : list,
                configs: dict
                ):
        self.id = id
        self.agents = agents
        self.environments = environments
        self.algorithm = algorithm
        self.data = None
        self.agentCount = 0
        self.configs = configs

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

    def save(self):
        shiva.save(self)

    def id_generator(self):
        id = self.agentCount
        self.agentCount +=1
        return id

###########################################################################
# 
#               Single Agent Learner
#         
###########################################################################

'''

    This Learner should be able to handle DQN and DDPG algorithms.

'''

class SingleAgentLearner(AbstractLearner):

    def __init__(self, 
                id,
                agents, 
                environments, 
                algorithm, 
                data, 
                configs
                ):

        super().__init__(
                        id,
                        agents, 
                        environments, 
                        algorithm, 
                        data, 
                        configs
                        )

        #I'm thinking about getting the saveFrequency here from the config and saving it in self
        self.saveFrequency = self.configs['Learner']['save_frequency']


    def run(self):

        for ep_count in range(self.configs['Learner']['episodes']):

            self.env.reset()

            self.totalReward = 0

            done = False
            while not done:
                done = self.step(ep_count)
                shiva.add_summary_writer(self, self.agents[0], 'Loss', self.alg.get_loss(), self.env.get_current_step())

        self.env.close()

    # Function to step throught the environment
    def step(self, ep_count):
       
        self.env.load_viewer()

        observation = self.env.get_observation()

        action = self.alg.get_action(self.agents[0], observation, self.env.get_current_step())

        next_observation, reward, done = self.env.step(action)

        # Write to tensorboard
        shiva.add_summary_writer(self, self.agents[0], 'Reward', reward, self.env.get_current_step())

        # Cumulate the reward
        self.totalReward += reward[0]

        self.buffer.append([observation, action, reward, next_observation, done])

        self.alg.update(self.agents[0], self.buffer.sample(), self.env.get_current_step())

        # when the episode ends
        if done:
            shiva.add_summary_writer(self, self.agents[0], 'Total Reward', self.totalReward, ep_count)
            shiva.add_summary_writer(self, self.agents[0], 'Average Loss per Episode', self.alg.get_average_loss(self.env.get_current_step()), ep_count)


        # Save the agent periodically
        # We need a criteria for this checkpoints
        if self.env.get_current_step() % self.saveFrequency == 0:
            pass

        return done

    def create_environment(self):
        # create the environment and get the action and observation spaces
        self.env = Environment.initialize_env(self.configs['Environment'])


    def get_agents(self):
        return self.agents

    def get_algorithm(self):
        return self.algorithm

    # Initialize the model
    def launch(self):

        # Launch the environment
        self.create_environment()

        # Launch the algorithm which will handle the
        self.alg = Algorithm.initialize_algorithm(self.env.get_observation_space(), self.env.get_action_space(), [self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])

        self.agents.append(self.alg.create_agent(self.id_generator()))
        
        self.buffer = ReplayBuffer.initialize_buffer(self.configs['ReplayBuffer'], 1, self.env.get_action_space(), self.env.get_observation_space())


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