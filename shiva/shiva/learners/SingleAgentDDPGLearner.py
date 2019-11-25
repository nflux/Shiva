from settings import shiva
from .Learner import Learner
import helpers.misc as misc
import envs
import algorithms
import buffers
import numpy as np
np.random.seed(5)

class SingleAgentDDPGLearner(Learner):
    def __init__(self, learner_id, config):
        super(SingleAgentDDPGLearner,self).__init__(learner_id, config)

    def run(self):
        self.step_count = 0
        for self.ep_count in range(self.episodes):
            self.env.reset()
            self.totalReward = 0
            self.kicks = 0
            self.turns = 0
            self.dashes = 0
            done = False
            while not done:
                done = self.step()
                self.step_count +=1

        self.env.close()

    def step(self):

        observation = self.env.get_observation()

        if self.manual_play:
            # This only works for users to play RoboCup!
            action = self.HPI.get_action(observation)
            while action is None:
                action = self.HPI.get_action(observation)
        else:
            action = self.alg.get_action(self.agent, observation, self.step_count)
        
        next_observation, reward, done, more_data = self.env.step(action, discrete_select='argmax')

        # TensorBoard Step Metrics
        shiva.add_summary_writer(self, self.agent, 'Actor Loss per Step', self.alg.get_actor_loss(), self.step_count)
        shiva.add_summary_writer(self, self.agent, 'Critic Loss per Step', self.alg.get_critic_loss(), self.step_count)
        shiva.add_summary_writer(self, self.agent, 'Normalized_Reward_per_Step', reward, self.step_count)
        shiva.add_summary_writer(self, self.agent, 'Raw_Reward_per_Step', more_data['raw_reward'], self.step_count)

        self.totalReward += more_data['raw_reward']
        print('to buffer:', observation.shape, more_data['action'].shape, reward.shape, next_observation.shape, [done])
        self.buffer.append([observation, more_data['action'].reshape(1,-1), reward, next_observation, int(done)])
        
        if self.step_count > self.alg.exploration_steps:
            # self.agent = 
            self.alg.update(self.agent, self.buffer.sample(), self.step_count)

        # Robocup actions
        if self.env.env_name == 'RoboCup':
            action = np.argmax(action[0:3])
            if action == 0:
                self.dashes += 1
            elif action == 1:
                self.turns += 1
            elif action == 2:
                self.kicks += 1
            shiva.add_summary_writer(self, self.agent, 'Action per Step', action, self.step_count)

            

        # Robocup Metrics
        if done and self.env.env_name == 'RoboCup':
            shiva.add_summary_writer(self, self.agent, 'Kicks per Episode', self.kicks, self.ep_count)
            shiva.add_summary_writer(self, self.agent, 'Turns per Episode', self.turns, self.ep_count)
            shiva.add_summary_writer(self, self.agent, 'Dashes per Episode', self.dashes, self.ep_count) 
            self.kicks = 0
            self.turns = 0
            self.dashes = 0

        # TensorBoard Episodic Metrics
        if done:
            shiva.add_summary_writer(self, self.agent, 'Total Reward per Episode', self.totalReward, self.ep_count)
            self.alg.ou_noise.reset()

            if self.ep_count % self.configs['Learner']['save_checkpoint_episodes'] == 0:
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

# class MetricsCalculator(object):
#     '''
#         Abstract class that it's solely purpose is to calculate metrics
#         Has access to the Environment
#     '''
#     def __init__(self, env, alg):
#         self.env = env
#         self.alg = alg
    
#     def Reward(self):
#         return self.env.get_reward()
        
#     def LossPerStep(self):
#         return self.alg.get_loss()

#     def LossActorPerStep(self):
#         return self.alg.get_actor_loss()

#     def TotalReward(self):
#         return self.get_total_reward()