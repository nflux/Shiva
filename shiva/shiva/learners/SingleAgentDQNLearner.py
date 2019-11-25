from settings import shiva
from .Learner import Learner
import helpers.misc as misc
import envs
import algorithms
import buffers
import numpy as np
import random

class SingleAgentDQNLearner(Learner):
    def __init__(self, learner_id, config):
        super(SingleAgentDQNLearner,self).__init__(learner_id, config)


    def run(self):
        self.step_count = 0
        for self.ep_count in range(self.episodes):
            self.env.reset()
            self.totalReward = 0
            done = False
            while not done:
                done = self.step()
                self.step_count +=1

        self.env.close()

    # Function to step throught the environment
    def step(self):

        observation = self.env.get_observation()

        action = self.alg.get_action(self.agent, observation, self.env.get_current_step())


        next_observation, reward, done, more_data = self.env.step(action)

        shiva.add_summary_writer(self, self.agent, 'Reward per Step', reward, self.step_count)
        shiva.add_summary_writer(self, self.agent, 'Loss per Step', self.alg.get_loss(), self.step_count)

        # Cumulate the reward
        self.totalReward += more_data['raw_reward'][0] if type(more_data['raw_reward']) == list else more_data['raw_reward']

        self.buffer.append([observation, action, reward, next_observation, done])

        self.alg.update(self.agent, self.buffer.sample(), self.step_count)

        # when the episode ends
        if done:
            shiva.add_summary_writer(self, self.agent, 'Total Reward per Episode', self.totalReward, self.ep_count)

        # Save the agent periodically
        # if self.step_count % self.save_frequency == 0:
        #     shiva._save_agent(self, self.agent)

        return done

    def create_environment(self):
        # create the environment and get the action and observation spaces
        environment = getattr(envs, self.configs['Environment']['type'])
        return environment(self.configs['Environment'])

    def create_algorithm(self):
        algorithm = getattr(algorithms, self.configs['Algorithm']['type'])
        try:
            self.configs['Agent']['learning_rate'] = random.uniform(self.learning_rate[0],self.learning_rate[1])
            return algorithm(self.env.get_observation_space(), self.env.get_action_space(),[self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])
        except:
            return algorithm(self.env.get_observation_space(), self.env.get_action_space(),[self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])
        
    def create_buffer(self):
        buffer = getattr(buffers,self.configs['Buffer']['type'])
        return buffer(self.configs['Buffer']['batch_size'], self.configs['Buffer']['capacity'])

    def get_agents(self):
        return self.agent

    def get_algorithm(self):
        return self.alg

    def launch(self):

        # Launch the environment
        self.env = self.create_environment()

        # Launch the algorithm which will handle the
        self.alg = self.create_algorithm()

        # Create the agent
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
