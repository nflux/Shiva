import numpy as np
import copy

from shiva.core.admin import Admin
from shiva.learners.Learner import Learner
from shiva.helpers.config_handler import load_class

class SingleAgentRoboDDPGLearner(Learner):
    def __init__(self, learner_id, config):
        super(SingleAgentRoboDDPGLearner,self).__init__(learner_id, config)
        np.random.seed(5)

    def run(self):
        self.step_count = 0
        for self.ep_count in range(self.episodes):
            self.env.reset()
            self.totalReward = 0
            self.kicks = 0
            self.kicked = 0
            self.turns = 0
            self.dashes = 0
            self.steps_per_episode = 0
            done = False
            while not done:
                done = self.step()
                self.step_count +=1
                self.steps_per_episode +=1

        self.env.close()

    def step(self):

        observation = self.env.get_observation()

        if self.ep_count == 10:
            self.manual_play = False 

        if self.manual_play:
            # This only works for users to play RoboCup!
            action = self.HPI.get_action(observation)
            while action is None:
                action = self.HPI.get_action(observation)
        else:
            action = self.alg.get_action(self.agent, observation, self.step_count)
        
        next_observation, reward, done, more_data = self.env.step(action) #, discrete_select='argmax')

        # TensorBoard Step Metrics
        Admin.add_summary_writer(self, self.agent, 'Actor_Loss_per_Step', self.alg.get_actor_loss(), self.step_count)
        Admin.add_summary_writer(self, self.agent, 'Critic_Loss_per_Step', self.alg.get_critic_loss(), self.step_count)
        # shiva.add_summary_writer(self, self.agent, 'Normalized_Reward_per_Step', reward, self.step_count)
        Admin.add_summary_writer(self, self.agent, 'Raw_Reward_per_Step', more_data['raw_reward'], self.step_count)

        # If ball was kicked
        if 150 < reward < 250:
            self.kicked += 1


        self.totalReward += more_data['raw_reward']

        # print('to buffer:', observation.shape, more_data['action'].shape, reward.shape, next_observation.shape, [done])
        # print('to buffer:', observation, more_data['action'], reward, next_observation, [done])

        t = [observation, more_data['action'].reshape(1,-1), reward, next_observation, int(done)]
        deep = copy.deepcopy(t)
        self.buffer.append(deep)
        
        if self.step_count > self.alg.exploration_steps:# and self.step_count % 16 == 0:
            self.alg.update(self.agent, self.buffer.sample(), self.step_count)
            # pass

        # Robocup actions
        if self.env.env_name == 'RoboCup':
            action = np.argmax(more_data['action'][:3])
            if action == 0:
                self.dashes += 1
            elif action == 1:
                self.turns += 1
            elif action == 2:
                self.kicks += 1
            Admin.add_summary_writer(self, self.agent, 'Action_per_Step', action, self.step_count)

            

        # Robocup Metrics
        if done and self.env.env_name == 'RoboCup':
            Admin.add_summary_writer(self, self.agent, 'Kicks_per_Episode', self.kicks, self.ep_count)
            Admin.add_summary_writer(self, self.agent, 'Turns_per_Episode', self.turns, self.ep_count)
            Admin.add_summary_writer(self, self.agent, 'Dashes_per_Episode', self.dashes, self.ep_count)
            Admin.add_summary_writer(self, self.agent, 'Steps_per_Episode', self.steps_per_episode, self.ep_count)
            Admin.add_summary_writer(self, self.agent, 'Ball_Kicks_per_Episode', self.kicked, self.ep_count)

            self.kicks = 0
            self.turns = 0
            self.dashes = 0
            self.kicked = 0

        # TensorBoard Episodic Metrics
        if done:
            Admin.add_summary_writer(self, self.agent, 'Total_Reward_per_Episode', self.totalReward, self.ep_count)
            self.alg.ou_noise.reset()

            if self.ep_count % self.configs['Learner']['save_checkpoint_episodes'] == 0:
                print("Checkpoint!")
                Admin.update_agents_profile(self)

        return done

    def create_environment(self):
        env_class = load_class('shiva.envs', self.configs['Environment']['type'])
        return env_class(self.configs['Environment'])

    def create_algorithm(self):
        algorithm_class = load_class('shiva.algorithms', self.configs['Algorithm']['type'])
        return algorithm_class(self.env.get_observation_space(), self.env.get_action_space(), [self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])
        
    def create_buffer(self):
        buffer_class = load_class('shiva.buffers', self.configs['Buffer']['type'])
        return buffer_class(self.configs['Buffer']['batch_size'], self.configs['Buffer']['capacity'])

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
            self.buffer = self._load_buffer(self.load_agents)
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
        return Admin._load_agents(path)[0]
