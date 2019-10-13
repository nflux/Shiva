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
    def __init__(self,
                agents,
                expert_agent,
                environments,
                algorithm,
                data,
                configs,
                learner_id,
                root):

        super().__init__(
                        agents,
                        environments,
                        algorithm,
                        data,
                        configs,
                        learner_id,
                        root)

        self.configs = configs[0]
        self.id = learner_id
        self.root = root


        #I'm thinking about getting the saveFrequency here from the config and saving it in self
        self.saveFrequency = configs[0]['Learner']['save_frequency']


    def run(self):
        self.supervised_update()
        self.imitation_update()




    def supervised_update(self):

        for ep_count in range(self.configs['Learner']['supervised_episodes']):

            self.env.reset()

            self.totalReward = 0

            done = False
            while not done:
                done = self.supervised_step(ep_count)
                self.writer.add_scalar('Loss', self.supervised_alg.get_loss(), self.env.get_current_step())

        # make an environment close function
        # self.env.close()
        self.env.env.close()

    def imitation_update(self):

        for ep_count in range(self.configs['Learner']['imitation_episodes']):

            self.env.reset()

            self.totalReward = 0

            done = False
            while not done:
                done = self.imitation_step(ep_count)
                self.writer.add_scalar('Loss', self.imitation_alg.get_loss(), self.env.get_current_step())

        self.env.env.close()



    # Function to step throught the environment
    def supervised_step(self,ep_count):
        self.env.load_viewer()

        observation = self.env.get_observation()

        action = self.supervised_alg.get_action(self.expert_agent, observation, self.env.get_current_step())

        next_observation, reward, done = self.env.step(action)

        # Write to tensorboard
        self.writer.add_scalar('Reward', reward, self.env.get_current_step())

        # Cumulate the reward
        self.totalReward += reward[0]

        self.expert_buffer.append([observation, action, reward, next_observation, done])

        self.supervised_alg.update(self.agents, self.expert_buffer.sample(), self.env.get_current_step())

        # when the episode ends
        if done:
            # add values to the tensorboard
            self.writer.add_scalar('Total Reward', self.totalReward, ep_count)
            self.writer.add_scalar('Average Loss per Episode', self.supervised_alg.get_average_loss(self.env.get_current_step()), ep_count)


        # Save the model periodically
        if self.env.get_current_step() % self.saveFrequency == 0:
            pass

        return done

    def imitation_step(self,ep_count):

        self.env.load_viewer()

        observation = self.env.get_observation()

        action = self.imitation_alg.get_action(self.agents[0],observation,self.env.get_current_step())

        next_observation, reward, done, = self.env.step(action)

        self.writer.add_scalar('Reward',reward,self.env.get_current_step())

        self.totalReward += reward[0]

        self.imitation_buffer.append([observation,action,reward,next_observation,done])

        self.imitation_alg.update(self.agents,self.expert_agent, self.imitation_buffer.sample(), self.env.get_current_step())

        # when the episode ends
        if done:
            # add values to the tensorboard
            self.writer.add_scalar('Total Reward', self.totalReward, ep_count)
            self.writer.add_scalar('Average Loss per Episode', self.imitation_alg.get_average_loss(self.env.get_current_step()), ep_count)


        # Save the model periodically
        if self.env.get_current_step() % self.saveFrequency == 0:
            pass

        return done




    def create_environment(self):
        # create the environment and get the action and observation spaces
        self.env = Environment.initialize_env(self.configs['Environment'])


    def get_agents(self):
        return self.agents[0]

    def get_algorithm(self):
        return self.algorithm

    # Initialize the model
    def launch(self):

        self.root = self.makeDirectory(self.root)

        # Launch the environment
        self.create_environment()

        # Launch the algorithm which will handle the
        self.supervised_alg,self.imitation_alg = Algorithm.initialize_algorithm(self.env.get_observation_space(), self.env.get_action_space(), [self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])
        #self.imitation_alg =  Algorithm.initialize_algorithm(self.env.get_observation_space(), self.env.get_action_space(), [self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])

        self.agents = self.supervised_alg.create_agent(self.root, self.id_generator())
        self.expert_agent = self.load_agent('Shiva/EliteAgents/MountainCar_DQAgent_300_57151.pth')

        log_dir = "{}/Agents/{}/logs".format(self.root, self.agents.id)

        print("\nHere's the directory to the tensorboard output\n",log_dir)

        self.writer =  SummaryWriter(log_dir)

        # Basic replay buffer at the moment
        self.expert_buffer = ReplayBuffer.initialize_buffer(self.configs['ReplayBuffer'], 1, self.env.get_action_space(), self.env.get_observation_space())
        self.imitation_buffer = ReplayBuffer.initialize_buffer(self.configs['ReplayBuffer'], 1, self.env.get_action_space(), self.env.get_observation_space())


    # do this for travis
    def load_agent(self, path):#,configs

        return Shiva._load_agents(expert_agent)


    def makeDirectory(self, root):

        # make the learner folder name
        root = root + '/learner{}'.format(self.id)

        # make the folder
        subprocess.Popen("mkdir " + root, shell=True)

        # return root for reference
        return root
