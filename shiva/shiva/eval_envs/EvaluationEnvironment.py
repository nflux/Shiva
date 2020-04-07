import gym

from shiva.envs.Environment import Environment

class EvaluationEnvironment(Environment):
    def __init__(self, configs):
        super(EvaluationEnvironment,self).__init__(configs)
        print('Config: ',self.configs['env_name'])
        self.env = gym.make(self.configs['env_name'])
        self.obs = self.env.reset()
        self.acs = 0
        self.rews = 0
        self.world_status = False
        self.configs = configs

    def step(self, action):
        pass

    def reset(self):
        pass

    def get_observation(self):
        pass

    def get_action(self):
        pass

    def get_reward(self):
        pass

    def load_viewer(self):
        pass

    def log(self, msg, to_print=False, verbose_level=-1):
        '''If verbose_level is not given, by default will log'''
        if verbose_level <= self.configs['Admin']['log_verbosity']['EvalEnv']:
            text = '{}\t{}'.format(str(self), msg)
            logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def close(self):
        pass
