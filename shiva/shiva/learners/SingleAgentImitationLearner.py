import torch
import numpy as np
import copy, socket, time, os, subprocess

from shiva.core.admin import Admin
from shiva.learners.Learner import Learner
from shiva.helpers.config_handler import load_class

class SingleAgentImitationLearner(Learner):
    def __init__(self,learner_id,config):
        super(SingleAgentImitationLearner, self).__init__(learner_id, config)
        torch.manual_seed(5)
        np.random.seed(5)

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
        self.step_count=0
        for self.ep_count in range(self.configs['Learner']['imitation_episodes']):
            self.env.reset()
            self.totalReward = 0
            done = False
            while not done:
                done = self.imitation_step(iter_count)
                self.step_count+=1
            self.env.close()

    # Function to step throught the environment
    def supervised_step(self):

        observation = self.env.get_observation()

        action = self.expert_agent.find_best_imitation_action(observation)

        #action = self.supervised_alg.get_action(self.expert_agent, observation)

        next_observation, reward, done, more_data = self.env.step(action)

        # Write to tensorboard
        Admin.add_summary_writer(self, self.expert_agent, 'Reward', reward, self.step_count)
        Admin.add_summary_writer(self, self.agent, 'Loss per step', self.imitation_alg.get_loss(),self.step_count)

        # Cumulate the reward
        self.totalReward += more_data['raw_reward'][0] if type(more_data['raw_reward']) == list else more_data['raw_reward']

        self.replay_buffer.append([observation, action, reward, next_observation, done])
        self.imitation_alg.supervised_update(self.agent,self.replay_buffer.sample(), self.step_count)

        # when the episode ends
        if done:
            # add values to the tensorboard
            Admin.add_summary_writer(self, self.agent, 'Total Reward', self.totalReward, self.ep_count)

            print(self.totalReward)

        return done

    def imitation_step(self,iter_count):

        #if iter_count == 4:
            #self.env.load_viewer()

        observation = self.env.get_observation()

        action = self.agent.find_best_action(observation)#, self.env.get_current_step())
        #action= torch.LongTensor(action)

        next_observation, reward, done, more_data = self.env.step(action)

        Admin.add_summary_writer(self, self.agent, 'Reward', reward, self.step_count)
        Admin.add_summary_writer(self, self.agent, 'Loss per step', self.imitation_alg.get_loss(), self.step_count)

        self.totalReward += more_data['raw_reward'][0] if type(more_data['raw_reward']) == list else more_data['raw_reward']

        self.replay_buffer.append([observation,action,reward,next_observation,done])
        self.imitation_alg.dagger_update(self.agent,self.expert_agent, self.replay_buffer.sample(), self.env.step_count)

        # when the episode ends
        if done:
            # add values to the tensorboard
            Admin.add_summary_writer(self, self.agent, 'Total Reward', self.totalReward, self.ep_count)
            print(self.totalReward)


        return done

    def create_environment(self):
        # create the environment and get the action and observation spaces
        environment = load_class('shiva.envs', self.configs['Environment']['type'])
        return environment(self.configs['Environment'])

    def create_algorithm(self):
        algorithm = load_class('shiva.algorithms', self.configs['Algorithm']['type'])
        acs_continuous = self.env.action_space_continuous
        acs_discrete= self.env.action_space_discrete
        return algorithm(self.env.get_observation_space(), self.env.get_action_space(),acs_discrete,acs_continuous,[self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])

    def create_buffer(self):
        buffer = load_class('shiva.buffers', self.configs['Buffer']['type'])
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
        self.imitation_alg = self.create_algorithm()

        self.agent = self.supervised_alg.create_agent()

        self.expert_agent = self.load_agent(self.configs['Learner']['expert_agent'])

        # Basic replay buffer at the moment
        if self.using_buffer:
            self.replay_buffer = self.create_buffer()

    def load_agent(self, path):#,configs

        return Admin._load_agents(path)[0]

    def makeDirectory(self, root):

        # make the learner folder name
        root = root + '/learner{}'.format(self.id)

        # make the folder
        subprocess.Popen("mkdir " + root, shell=True)

        # return root for reference
        return root

class SingleAgentRoboCupImitationLearner(Learner):
    def __init__(self,learner_id,config,port=None):
        super(SingleAgentRoboCupImitationLearner, self).__init__(learner_id,config)
        self.supervised_episodes = self.episodes*self.percent_ep_super
        self.imitation_episodes = self.episodes-self.supervised_episodes
        self.port = port
        self.step_count = 0
        self.ep_count = 0
        # self.train = torch.BoolTensor([True]).share_memory_()
        # self.lr_decay = (self.configs['Agent']['learning_rate']-self.configs['Agent']['lr_end'])/self.configs['Learner']['imitation_episodes']
        

    def run(self):
        self.supervised_update()
        self.imitation_update()
        self.env.close()

    def send_imit_obs_msgs(self):
        self.comm.send(self.env.get_imit_obs_msg())

    def recv_imit_acs_msgs(self):
        while True:
            acs_msg = self.comm.recv(8192)
            if acs_msg != b'':
                break

        acs_msg = str(acs_msg).split(' ')[1:-1]

        return np.array(list(map(lambda x: float(x), acs_msg)))

    def supervised_update(self):
        # while self.ep_count < self.supervised_episodes:

        for _ in range(self.supervised_episodes):
            self.ep_count += 1
            self.env.reset()
            done = False
            while not done:
                done = self.supervised_step()
                self.step_count +=1

    def imitation_update(self):
        for _ in range(self.imitation_episodes):
            self.ep_count += 1
            self.env.reset()
            done = False
            while not done:
                done = self.imitation_step()
                self.step_count+=1
            self.env.close()

    # Function to step throught the environment
    def supervised_step(self):

        observation = self.env.get_observation()

        self.send_imit_obs_msgs()
        action = self.recv_imit_acs_msgs()

        next_observation, reward, done, more_data = self.env.step(action, discrete_select='argmax')

        # Write to tensorboard
        # Admin.add_summary_writer(self, self.agent, 'Reward', reward, self.step_count)
        # Admin.add_summary_writer(self, self.agent, 'Loss_per_step', self.imitation_alg.get_loss(),self.step_count)

        # Cumulate the reward
        self.totalReward += more_data['raw_reward'][0] if type(more_data['raw_reward']) == list else more_data['raw_reward']

        self.super_buffer.push(copy.deepcopy([torch.from_numpy(observation), torch.from_numpy(action), torch.from_numpy(reward),
                                                torch.from_numpy(next_observation), torch.from_numpy(np.array([done])).float()]))
        # self.replay_buffer.append(copy.deepcopy([observation, action, reward, next_observation, done, None]))
        self.imitation_alg.supervised_update(self.agent,self.super_buffer.sample(self.imitation_alg.device), self.step_count)

        # when the episode ends
        # if done:
        #     # add values to the tensorboard
        #     Admin.add_summary_writer(self, self.agent, 'Total_Reward', self.totalReward, self.ep_count)

        return done

    def imitation_step(self):

        observation = self.env.get_observation()

        self.send_imit_obs_msgs()
        bot_action = self.recv_imit_acs_msgs()
        action = self.agent.find_best_imitation_action(observation)

        next_observation, reward, done, more_data = self.env.step(action)

        # Admin.add_summary_writer(self, self.agent, 'Reward', reward, self.step_count)
        # Admin.add_summary_writer(self, self.agent, 'Loss_per_step', self.imitation_alg.get_loss(), self.step_count)

        # self.totalReward += more_data['raw_reward'][0] if type(more_data['raw_reward']) == list else more_data['raw_reward']

        self.buffer.push(copy.deepcopy([torch.from_numpy(observation),torch.from_numpy(more_data['action'].reshape(1,-1)),
                                                torch.from_numpy(reward),torch.from_numpy(next_observation),torch.from_numpy(np.array([done])).float(),
                                                torch.from_numpy(bot_action)]))
        self.imitation_alg.dagger_update(self.agent, self.buffer.sample(self.imitation_alg.device), self.step_count)

        # when the episode ends
        # if done:
            # # add values to the tensorboard
            # Admin.add_summary_writer(self, self.agent, 'Total_Reward', self.totalReward, self.ep_count)
            # self.totalGoals += self.env.isGoal()
            # Admin.add_summary_writer(self, self.agent, 'Goal_Percentage', (self.totalGoals/(self.ep_count+1))*100.0, self.ep_count)
            # # if self.configs['Admin']['save'] and (self.ep_count % self.configs['Learner']['save_frequency'] == 0):
            # #     Admin._save_agent(self, self.agent)
            
            # # if self.ep_count % self.configs['Agent']['lr_change_every'] == 0:
            # #     self.agent.learning_rate = self.configs['Agent']['learning_rate'] - (self.ep_count * self.lr_decay)
            # #     print('lr', str(self.agent.learning_rate))
            # #     self.agent.mod_lr(self.agent.actor_optimizer, self.agent.learning_rate)

        return done

    def create_environment(self):
        env_class = load_class('shiva.envs', self.configs['Environment']['type'])
        return env_class(self.configs['Environment'], self.port+4)

    def create_algorithm(self):
        algorithm = load_class('shiva.algorithms', self.configs['Algorithm']['type'])
        return algorithm(self.env.get_observation_space(), self.env.get_action_space(),[self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])

    def create_buffers(self, obs_dim, ac_dim):
        buffer1 = load_class('shiva.buffers', self.configs['Buffer']['type1'])
        buffer2 = load_class('shiva.buffers', self.configs['Buffer']['type2'])
        return (buffer1(self.configs['Buffer']['super_capacity'], self.configs['Buffer']['super_batch_size'], 1, obs_dim, ac_dim),
                buffer2(self.configs['Buffer']['dagger_capacity'], self.configs['Buffer']['dagger_batch_size'], 1, obs_dim, ac_dim))

    def get_agent(self):
        return self.agent

    def get_algorithm(self):
        return self.algorithm

    # Initialize the model
    def launch(self):

        # Launch the environment
        self.env = self.create_environment()
        self.env.launch()

        # Launch the algorithm which will handle the
        self.imitation_alg = self.create_algorithm()

        self.agent = self.imitation_alg.create_agent()

        # Basic replay buffer at the moment
        if self.using_buffer:
            self.super_buffer, self.buffer = self.create_buffers(self.env.observation_space, self.env.action_space['discrete'] + self.env.action_space['param'])
        
        cmd = [os.getcwd() + '/shiva/envs/RoboCupBotEnv.py', '-p', str(self.port)]
        self.bot_process = subprocess.Popen(cmd, shell=False)

        while True:
            try:
                self.comm = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.comm.connect(('127.0.0.1', self.port+3))
                break
            except:
                time.sleep(0.1)

    def close(self):
        try:
            self.bot_process.terminate()
            time.sleep(0.1)
            self.bot_process.kill()
        except:
            pass

    def load_agent(self, path):#,configs

        return Admin._load_agents(path)[0]

    def makeDirectory(self, root):

        # make the learner folder name
        root = root + '/learner{}'.format(self.id)

        # make the folder
        subprocess.Popen("mkdir " + root, shell=True)

        # return root for reference
        return root