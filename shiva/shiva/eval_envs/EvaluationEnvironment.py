from envs.Environment import Environment
import gym

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

    def close(self):
        pass
