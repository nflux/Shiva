class Environment:
    def __init__(self, configs):
        {setattr(self, k, v) for k,v in configs.items()}
        self.configs = configs

    def step(self,actions):
        pass

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

    def reset(self):
        pass

    def load_viewer(self):
        pass

    def normalize_reward(self):
        return (self.b-self.a)*(self.rews-self.min)/(self.max-self.min)