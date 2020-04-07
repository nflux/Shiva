import torch

class Environment:
    def __init__(self, configs):
        if 'MetaLearner' in configs:
            {setattr(self, k, v) for k,v in configs['Environment'].items()}
        else:
            {setattr(self, k, v) for k,v in configs.items()}
        self.configs = configs
        self.steps_per_episode = 0
        self.step_count = 0
        self.done_count = 0
        self.total_episodes_to_play = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def step(self,actions):
        pass

    def finished(self, n_episodes=None):
        '''
            The environment controls when the Learner is done stepping on the Environment
            Here we evaluate the amount of episodes to be played
        '''
        assert (n_episodes is not None), 'A @n_episodes is required to check if we are done running the Environment'
        return self.done_count >= n_episodes

    def start_env(self):
        return True

    def get_observation(self, agent):
        pass

    def get_observations(self):
        pass

    def get_action(self, agent):
        pass

    def get_actions(self):
        pass

    def get_reward(self, agent):
        pass

    def get_rewards(self):
        pass

    def get_observation_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space

    def get_current_step(self):
        return self.step_count

    def get_metrics(self):
        '''
            To be implemented per Environment
        '''
        pass

    def reset(self):
        pass

    def load_viewer(self):
        pass

    def normalize_reward(self, reward):
        return self.reward_factor*(reward-self.min_reward)/(self.max_reward-self.min_reward)

    def log(self, msg, to_print=False, verbose_level=-1):
        '''If verbose_level is not given, by default will log'''
        if verbose_level <= self.configs['Admin']['log_verbosity']['Env']:
            text = "{}\t\t\t{}".format(str(self), msg)
            logger.info(text, to_print or self.configs['Admin']['print_debug'])
