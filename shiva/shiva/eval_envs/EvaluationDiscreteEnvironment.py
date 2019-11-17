import numpy as np
from envs import Environment

def initialize_eval_env(env_params):
    assert 'env_type' in env_params, 'Needs @env_type in [EVAL_ENV] config'
    
    if env_params['env_type'] == 'Gym':
        return EvaluationDiscreteEnvironment(env_params['environment'], env_params['metrics_strings'], env_params['env_render'])

class EvaluationDiscreteEnvironment(Environment):
    def __init__(self, eval_config, env_config):
        self.metrics_strings = metrics_strings
        self.reset_metrics()
        self.metric_calc = MetricsCalculator(self)
        super(EvaluationDiscreteEnvironment, self).__init__(env_config)

    def __str__(self):
        return "<GymEvaluationEnvironment:{}>".format(self.env_name)

    def reset(self):
        '''
            Reset the environment to start a new episode
            Updates the episodic metrics
        '''
        # self._update_metrics_per_episode()
        super(GymEvaluationEnvironment, self).reset()
        self.rewards_per_step = []

    def reset_metrics(self):
        '''
            This method is to reinitialize all the metrics
        '''
        self.episodes_count = 0
        self.rewards_per_step = []
        self.rewards_per_episode, self.steps_per_episode = [], []

    def step(self, action):
        next_observation, reward, done = super(GymEvaluationEnvironment, self).step(action)
        self._update_metrics_per_step()
        if done:
            self._update_metrics_per_episode()
        return next_observation, reward, done

    def _update_metrics_per_step(self):
        '''
            Metrics to be recorded per step
        '''
        self.rewards_per_step.append(self.get_reward())

    def _update_metrics_per_episode(self):
        '''
            Metrics to be recorded per episode
        '''
        self.episodes_count += 1
        self.rewards_per_episode.append(np.sum(self.rewards_per_step))
        self.steps_per_episode.append(self.get_current_step())

    def get_metrics(self, reset_metrics=True):
        '''
            Calculates all the metrics given in the config file

            Input
                @reset_metrics       will reset the episodic metrics, most common to be True

            Return
                List of metric tuples
                    g.e. [ ('cumulative_reward', 100), (...), (...) ]
        '''
        _return = []
        print('\t# episodes: {}'.format(self.episodes_count))
        # print('\tRewards per episode: {}\n\tSteps per episode: {}'.format(str(self.rewards_per_episode), str(self.steps_per_episode)))
        for metric_name in self.metrics_strings:
            val = getattr(self.metric_calc, metric_name)()
            _return.append( (metric_name, val) )
            print("\t{}: {}".format(metric_name, val))
        if reset_metrics:
            self.reset_metrics()
        return _return


class MetricsCalculator(object):
    '''
        Abstract class that it's solely purpose is to calculate metrics
        Has access to the Environment
    '''
    def __init__(self, env):
        self.env = env
    
    def AveRewardPerEpisode(self):
        return np.average(self.env.rewards_per_episode)

    def MaxEpisodicReward(self):
        return np.max(self.env.rewards_per_episode)

    def MinEpisodicReward(self):
        return np.min(self.env.rewards_per_episode)

    def AveStepsPerEpisode(self):
        return np.average(self.env.steps_per_episode)