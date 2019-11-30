from settings import shiva
from .Learner import Learner
import helpers.misc as misc
import envs
import algorithms
import buffers
from copy import deepcopy

class UnityLearner(Learner):
    def __init__(self, learner_id, config):
        super(UnityLearner,self).__init__(learner_id, config)
        self.done_counter = 0

    # so now the done is coming from the environment
    def run(self):
        self.step_count = 0

        while self.continue_run():
            self.exploration_mode = self.step_count < self.alg.exploration_steps
            self.step()
        self.env.close()

    def continue_run(self):
        return (self.episodes > 0 and self.env.done_counts < self.episodes) or (self.steps > 0 and self.step_count < self.steps)

    def step(self):
        self.step_count += 1
        observation = self.env.get_observation()
        # print("Learner:", observation.shape)
        # action = self.env.get_random_action()
        action = [self.alg.get_action(self.agent, obs, self.step_count) for obs in observation]

        # print("Learner:", observation.shape)

        # print("Learner:", action)

        print('step:', self.step_count, '\treward:', self.env.reward_total)

        next_observation, reward, done, _ = self.env.step(action)
        for obs, act, rew, next_obs, don in zip(observation, action, reward, next_observation, done):
            exp = [obs, act, rew, next_obs, int(don)]
            exp = deepcopy(exp)
            self.buffer.append(exp)
        
        if not self.exploration_mode and self.step_count % 8 == 0:
            self.alg.update(self.agent, self.buffer.sample(), self.step_count)

        # self.add_summary_writer(self, self.agent, 'Total Reward', self.env.reward_total, self.step_count)
        
        # input()
        
        # TensorBoard metrics
        # shiva.add_summary_writer(self, self.agent, 'Actor Loss per Step', self.alg.get_actor_loss(), self.step_count)
        # shiva.add_summary_writer(self, self.agent, 'Critic Loss per Step', self.alg.get_critic_loss(), self.step_count)
        # # shiva.add_summary_writer(self, self.agent, 'Normalized_Reward_per_Step', reward, self.step_count)
        # # shiva.add_summary_writer(self, self.agent, 'Raw_Reward_per_Step', more_data['raw_reward'], self.step_count)
        # shiva.add_summary_writer(self, self.agent, 'Action in Z per Step', action[0], self.step_count)
        # shiva.add_summary_writer(self, self.agent, 'Action in X per Step', action[1], self.step_count)

        # self.totalReward += reward  # more_data['raw_reward']

        # t = [observation, action, reward, next_observation, int(done)]

        # deep = deepcopy(t)

        # print(self.ep_count)
        # print('to buffer:', t)

        # self.buffer.append(deep)

        # if self.step_count > self.alg.exploration_steps and self.step_count % 16 == 0:
        #     self.agent = self.alg.update(self.agent, self.buffer.sample(), self.step_count)
            # self.alg.update(self.agent, self.buffer.sample(), self.step_count)

        # TensorBoard Metrics
        # if done:
        #     shiva.add_summary_writer(self, self.agent, 'Ave Reward per Episode', self.env.aver_rew, self.done_counter)
        #     # self.alg.ou_noise.reset()
        #     self.done_counter += 1
            
        # if self.step_count % 100 == 0:
        #     self.alg.ou_noise.reset()

        # No need to return anything
        # return True

    def create_environment(self):
        # create the environment and get the action and observation spaces
        environment = getattr(envs, self.configs['Environment']['type'])
        return environment(self.configs['Environment'])

    def create_algorithm(self):
        algorithm = getattr(algorithms, self.configs['Algorithm']['type'])
        return algorithm(self.env.observation_space, self.env.action_space, [self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])
        
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

        # # Launch the algorithm which will handle the
        self.alg = self.create_algorithm()

        # # Create the agent
        if self.configs['Learner']['load_agents'] is not False:
            self.agent = self.load_agent(self.configs['Learner']['load_agents'])
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
