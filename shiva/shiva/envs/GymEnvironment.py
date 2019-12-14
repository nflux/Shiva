import gym
import numpy as np
from .Environment import Environment

class GymEnvironment(Environment):
    def __init__(self, configs):
        super(GymEnvironment,self).__init__(configs)
        self.env = gym.make(self.env_name)
        self.obs = self.env.reset()

        self.done = False
        self.action_space_continuous = None
        self.action_space_discrete = None
        self.observation_space = self.set_observation_space()
        self.action_space = self.set_action_space()

        self.steps_per_episode = 0
        self.step_count = 0
        self.done_count = 0
        self.reward_per_step = 0
        self.reward_per_episode = 0
        self.reward_total = 0

        self.render = configs['render']

    def step(self, action):
        self.acs = action
        action4Gym = np.argmax(action) if self.action_space_continuous is None else action
        self.obs, self.reward_per_step, self.done, info = self.env.step(action4Gym)
        self.load_viewer()
        '''
            Metrics collection
                Episodic # of steps             self.steps_per_episode --> is equal to the amount of instances on Unity, 1 Shiva step could be a couple of Unity steps
                Cumulative # of steps           self.step_count
                Cumulative # of episodes        self.done_count
                Step Reward                     self.reward_per_step
                Episodic Reward                 self.reward_per_episode
                Cumulative Reward               self.reward_total
        '''
        self.steps_per_episode += 1
        self.step_count += 1
        self.done_count += 1 if self.done else 0
        # self.reward_per_step = self.reward_per_step
        self.reward_per_episode += self.reward_per_step
        self.reward_total += self.reward_per_step

        if self.normalize:
            return self.obs, self.normalize_reward(self.reward_per_step), self.done, {'raw_reward': self.reward_per_step, 'action': action}
        else:
            return self.obs, self.reward_per_step, self.done, {'raw_reward': self.reward_per_step, 'action': action}

    def reset(self):
        self.steps_per_episode = 0
        self.reward_per_step = 0
        self.reward_per_episode = 0
        self.done = False
        self.obs = self.env.reset()

    def get_metrics(self, episodic=False):
        if not episodic:
            metrics = [
                ('Reward/Per_Step', self.reward_per_step)
            ]
        else:
            metrics = [
                ('Reward/Per_Episode', self.reward_per_episode),
                ('Agent/Steps_Per_Episode', self.steps_per_episode)
            ]
        return metrics

    def is_done(self):
        return self.done

    def set_observation_space(self):
        observation_space = 1
        if self.env.observation_space.shape != ():
            for i in range(len(self.env.observation_space.shape)):
                observation_space *= self.env.observation_space.shape[i]
        else:
            observation_space = self.env.observation_space.n

        return observation_space

    def set_action_space(self):
        action_space = 1
        if self.env.action_space.shape != ():
            '''
                Portion where Action Space is Continuous
            '''
            for i in range(len(self.env.action_space.shape)):
                action_space *= self.env.action_space.shape[i]
            self.action_space_continuous = action_space
            # self.action_space = action_space
        else:
            '''
                Portion where Action Space is Discrete
            '''
            action_space = self.env.action_space.n
            self.action_space_discrete = action_space
        return action_space

    def get_observation(self):
        return self.obs

    def get_action(self):
        return self.acs

    def get_reward(self):
        return self.reward_per_step

    def get_total_reward(self):
        '''
            Returns episodic reward
        '''
        return self.reward_per_episode

    def load_viewer(self):
        if self.render:
            self.env.render()

    def close(self):
        self.env.close()
