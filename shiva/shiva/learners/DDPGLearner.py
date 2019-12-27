from copy import deepcopy

from shiva.core.admin import Admin
from shiva.learners.Learner import Learner
from shiva.helpers.config_handler import load_class

class DDPGLearner(Learner):
    def __init__(self, learner_id, config):
        super(DDPGLearner,self).__init__(learner_id, config)
        self.done_counter = 0

    # so now the done is coming from the environment
    def run(self):
        self.step_count = 0
        for self.ep_count in range(self.episodes):
            # self.env.reset()
            self.totalReward = 0
            done = False
            while not done:
                done = self.step()
                self.step_count +=1

        self.env.close()

    def step(self):

        observation = self.env.get_observation()
        
        action = self.alg.get_action(self.agent, observation, self.step_count)

        # print("Learner:", observation)

        # print("Learner:", action)

        next_observation, reward, done, more_data = self.env.step(action)

        # TensorBoard metrics
        Admin.add_summary_writer(self, self.agent, 'Actor Loss per Step', self.alg.get_actor_loss(), self.step_count)
        Admin.add_summary_writer(self, self.agent, 'Critic Loss per Step', self.alg.get_critic_loss(), self.step_count)
        # shiva.add_summary_writer(self, self.agent, 'Normalized_Reward_per_Step', reward, self.step_count)
        # shiva.add_summary_writer(self, self.agent, 'Raw_Reward_per_Step', more_data['raw_reward'], self.step_count)
        Admin.add_summary_writer(self, self.agent, 'Action in Z per Step', action[0], self.step_count)
        Admin.add_summary_writer(self, self.agent, 'Action in X per Step', action[1], self.step_count)

        self.totalReward += reward  # more_data['raw_reward']

        t = [observation, action, reward, next_observation, int(done)]

        deep = deepcopy(t)

        # print(self.ep_count)
        # print('to buffer:', t)

        self.buffer.append(deep)

        if self.step_count > self.alg.exploration_steps and self.step_count % 16 == 0:
            self.agent = self.alg.update(self.agent, self.buffer.sample(), self.step_count)
            # self.alg.update(self.agent, self.buffer.sample(), self.step_count)

        # TensorBoard Metrics
        if done:
            Admin.add_summary_writer(self, self.agent, 'Ave Reward per Episode', self.env.aver_rew, self.done_counter)
            # self.alg.ou_noise.reset()
            self.done_counter += 1
            
        if self.step_count % 100 == 0:
            self.alg.ou_noise.reset()

        return done

    def create_environment(self):
        env_class = load_class('shiva.envs', self.configs['Environment']['type'])
        return env_class(self.configs)

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

        # Launch the algorithm which will handle the
        self.alg = self.create_algorithm()

        # Create the agent
        if self.configs['Learner']['load_agents'] is not False:
            self.agent = self.load_agent(self.configs['Learner']['load_agents'])
        else:
            self.agent = self.alg.create_agent()
        
        # if buffer set to true in config
        if self.using_buffer:
            # Basic replay buffer at the moment
            self.buffer = self.create_buffer()

        print('Launch Successful.')


    def save_agent(self):
        pass

    def load_agent(self, path):
        return Admin._load_agents(path)[0]
