
from settings import shiva
from .Learner import Learner
import helpers.misc as misc
import envs
import algorithms
import buffers
import torch

class SingleAgentImitationLearner(Learner):
    def __init__(self,learner_id,config):

        super(SingleAgentImitationLearner, self).__init__(learner_id,config)




    def run(self):

        self.supervised_update()
        self.imitation_update()
        self.env.close()

    def supervised_update(self):
        self.step_count = 0
        for self.ep_count in range(self.configs['Learner']['supervised_episodes']):
            self.env.reset()
            self.totalReward = 0
            done = False
            while not done:
                done = self.supervised_step()
                self.step_count +=1


        # make an environment close function
        # self.env.close()
        self.env.close()
        self.agent = self.agents[0]

        #self.supervised_train()

    def imitation_update(self):
        for iter_count in range(1,self.configs['Learner']['dagger_iterations']):

            self.step_count=0
            for self.ep_count in range(self.configs['Learner']['imitation_episodes']):
                self.env.reset()
                self.totalReward = 0
                done = False
                while not done:
                    #next_observation, reward, done, _ = self.env.step([0.0,1.0,0.0,0.0])
                    done = self.imitation_step(iter_count)
                    self.step_count+=1
                self.env.close()
            self.agent = self.agents[iter_count]


    # Function to step throught the environment
    def supervised_step(self):

        observation = self.env.get_observation()

        action = self.expert_agent.find_best_imitation_action(observation)

        #action = self.supervised_alg.get_action(self.expert_agent, observation)

        next_observation, reward, done, more_data = self.env.step(action)

        # Write to tensorboard
        shiva.add_summary_writer(self, self.expert_agent, 'Reward', reward, self.step_count)
        shiva.add_summary_writer(self, self.agents[0], 'Loss_per_step', self.supervised_alg.get_loss(),self.step_count)

        # Cumulate the reward
        self.totalReward += more_data['raw_reward'][0] if type(more_data['raw_reward']) == list else more_data['raw_reward']

        self.replay_buffer.append([observation, action, reward, next_observation, done])
        self.supervised_alg.update(self.agents[0],self.replay_buffer.sample(), self.step_count)

        # when the episode ends
        if done:
            # add values to the tensorboard
            shiva.add_summary_writer(self, self.agents[0], 'Total_Reward', self.totalReward, self.ep_count)

            print(self.totalReward)


        return done

    def imitation_step(self,iter_count):

        #if iter_count == 4:
            #self.env.load_viewer()

        observation = self.env.get_observation()

        action = self.agents[iter_count-1].find_best_action(self.agents[iter_count-1].policy, observation)#, self.env.get_current_step())
        #action= torch.LongTensor(action)

        next_observation, reward, done, more_data = self.env.step(action)


        shiva.add_summary_writer(self, self.agents[0], 'Reward', reward, self.step_count)
        shiva.add_summary_writer(self, self.agents[0], 'Loss_per_step', self.imitation_alg.get_loss(), self.step_count)


        self.totalReward += more_data['raw_reward'][0] if type(more_data['raw_reward']) == list else more_data['raw_reward']

        self.replay_buffer.append([observation,action,reward,next_observation,done])
        self.imitation_alg.update(self.agents[iter_count],self.expert_agent, self.replay_buffer.sample(), self.env.step_count)


        #print('Total Reward: ', self.totalReward)
        #print('Average Loss per Episode', self.supervised_alg.get_average_loss(self.env.get_current_step()))
        # when the episode ends
        if done:
            # add values to the tensorboard
            shiva.add_summary_writer(self, self.agents[0], 'Total_Reward', self.totalReward, self.ep_count)
            print(self.totalReward)


        return done


    def create_environment(self):
        # create the environment and get the action and observation spaces
        environment = getattr(envs, self.configs['Environment']['type'])
        return environment(self.configs['Environment'])

    def create_algorithm(self):
        algorithm = getattr(algorithms, self.configs['Algorithm']['type1'])
        algorithm2 = getattr(algorithms, self.configs['Algorithm']['type2'])
        acs_continuous = self.env.action_space_continuous
        acs_discrete= self.env.action_space_discrete
        return algorithm(self.env.get_observation_space(), self.env.get_action_space(),acs_discrete,acs_continuous,[self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']]), algorithm2(self.env.get_observation_space(), self.env.get_action_space(),acs_discrete,acs_continuous,[self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])


    def create_buffer(self):
        buffer = getattr(buffers,self.configs['Buffer']['type'])
        return buffer(self.configs['Buffer']['batch_size'], self.configs['Buffer']['capacity'])


    def get_agents(self):
        return self.agents

    def get_algorithm(self):
        return self.algorithm

    # Initialize the model
    def launch(self):



        # Launch the environment
        self.env = self.create_environment()


        # Launch the algorithm which will handle the
        self.supervised_alg,self.imitation_alg = self.create_algorithm()
        #Algorithm.initialize_algorithm(self.env.get_observation_space(), self.env.get_action_space(), [self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])
        #self.imitation_alg =  Algorithm.initialize_algorithm(self.env.get_observation_space(), self.env.get_action_space(), [self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])

        self.agents = [None] * self.configs['Learner']['dagger_iterations']
        for i in range(len(self.agents)):
            self.agents[i] = self.supervised_alg.create_agent()
        self.agent = self.agents[0]

        self.expert_agent = self.load_agent(self.configs['Learner']['expert_agent'])

        # Basic replay buffer at the moment
        if self.using_buffer:
            self.replay_buffer = self.create_buffer()
        #self.imitation_buffer = ReplayBuffer.initialize_buffer(self.configs['ReplayBuffer'], 1, self.env.get_action_space(), self.env.get_observation_space())


    # do this for travis
    def load_agent(self, path):#,configs

        return shiva._load_agents(path)[0]


    def makeDirectory(self, root):

        # make the learner folder name
        root = root + '/learner{}'.format(self.id)

        # make the folder
        subprocess.Popen("mkdir " + root, shell=True)

        # return root for reference
        return root
