from __main__ import shiva
from .Learner import Learner
import helpers.misc as misc
import envs
import algorithms
import buffers
import numpy as np
np.random.seed(5)
import copy

class SingleAgentDDPGLearner(Learner):
    def __init__(self, learner_id, config):
        super(SingleAgentDDPGLearner,self).__init__(learner_id, config)

    def run(self):
        self.step_count = 0
        for self.ep_count in range(self.episodes):
            self.env.reset()
            self.totalReward = 0
            self.steps_per_episode = 0
            done = False
            while not done:
                done = self.step()
                self.step_count +=1
                self.steps_per_episode +=1

        self.env.close()

    def step(self):

        observation = self.env.get_observation()

        action = self.alg.get_action(self.agent, observation, self.step_count)
        
        next_observation, reward, done, more_data = self.env.step(action) #, discrete_select='argmax')

        # TensorBoard Step Metrics
        shiva.add_summary_writer(self, self.agent, 'Actor_Loss_per_Step', self.alg.get_actor_loss(), self.step_count)
        shiva.add_summary_writer(self, self.agent, 'Critic_Loss_per_Step', self.alg.get_critic_loss(), self.step_count)
        # shiva.add_summary_writer(self, self.agent, 'Normalized_Reward_per_Step', reward, self.step_count)
        shiva.add_summary_writer(self, self.agent, 'Raw_Reward_per_Step', more_data['raw_reward'], self.step_count)

        self.totalReward += more_data['raw_reward']

        # print('to buffer:', observation.shape, more_data['action'].shape, reward.shape, next_observation.shape, [done])
        # print('to buffer:', observation, more_data['action'], reward, next_observation, [done])

        t = [observation, more_data['action'].reshape(1,-1), reward, next_observation, int(done)]
        deep = copy.deepcopy(t)
        self.buffer.append(deep)
        
        if self.step_count > self.alg.exploration_steps:# and self.step_count % 16 == 0:
            self.agent = self.alg.update(self.agent, self.buffer.sample(), self.step_count)
            # pass

        # TensorBoard Episodic Metrics
        if done:
            shiva.add_summary_writer(self, self.agent, 'Total_Reward_per_Episode', self.totalReward, self.ep_count)
            self.alg.ou_noise.reset()

            if self.ep_count % self.configs['Learner']['save_checkpoint_episodes'] == 0:
                print("Checkpoint!")
                shiva.update_agents_profile(self)

        return done

    def create_environment(self):
        # create the environment and get the action and observation spaces
        environment = getattr(envs, self.configs['Environment']['type'])
        return environment(self.configs['Environment'])

    def create_algorithm(self):
        algorithm = getattr(algorithms, self.configs['Algorithm']['type'])
        return algorithm(self.env.get_observation_space(), self.env.get_action_space(), [self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])

    def create_buffer(self):
        buffer = getattr(buffers,self.configs['Buffer']['type'])
        return buffer(self.configs['Buffer']['batch_size'], self.configs['Buffer']['capacity'])

    def get_agents(self):
        return self.agents

    def get_algorithm(self):
        return self.alg

    def launch(self):

        # Launch the environment
        self.env = self.create_environment()

        if self.manual_play:
            self.HPI = envs.HumanPlayerInterface()


        # Launch the algorithm which will handle the
        self.alg = self.create_algorithm()

        # Create the agent
        if self.load_agents:
            self.agent = self.load_agent(self.load_agents)
            # self.buffer = self._load_buffer(self.load_agents)
        else:
            self.agent = self.alg.create_agent(self.get_id())

        # if buffer set to true in config
        if self.using_buffer:
            # Basic replay buffer at the moment
            self.buffer = self.create_buffer()

        print('Launch Successful.')


    def save_agent(self):
        pass

    def load_agent(self, path):
        return shiva._load_agents(path)[0]
