import gym
import numpy as np
from envs.Environment import Environment
from .Evaluation import Evaluation
from .GymDiscreteEvaluationEnvironment import GymDiscreteEvaluationEnvironment

class GymDiscreteEvaluation(Evaluation):
    def __init__(self,
                configs: 'whole config passed',
                learners
    ):
        {setattr(self, k, v) for k,v in configs.items()}
        self.configs = configs
        self.learners = learners
        self.eval_envs = [None] * len(learners)
        self.eval_scores = [None] * len(learners)
        self._create_eval_envs()

    def evaluate_agents(self):
        '''
            Starts evaluation process
            This implementation is specific to each environment type
        '''
        episode_scores = np.zeros(self.configs['episodes'])
        for i in range(len(self.learners)):
            for e in range(self.configs['episodes']):
                self.eval_envs[i].reset()
                self.totalReward = 0
                done = False
                while not done:
                    done = self.step(i)
                episode_scores[e] = self.totalReward
            self.eval_scores[i] = episode_scores

            self.eval_envs[i].close()


    def _create_eval_envs(self):
        '''
            This implementation is specific to each environment type
        '''
        for i in range(len(self.eval_envs)):
            self.eval_envs[i] = GymDiscreteEvaluationEnvironment(self.configs)

    def rank_agents(self, validation_scores):
        pass

    def step(self,idx):

        observation = self.eval_envs[idx].get_observation()

        action = self.learners[idx].agent.get_action(observation)

        next_observation, reward, done, more_data = self.eval_envs[idx].step(action)

        # Cumulate the reward
        self.totalReward += reward[0]

        return done
