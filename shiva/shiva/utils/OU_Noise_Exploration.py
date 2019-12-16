from utils.Noise import OUNoise
from utils.Base_Exploration_Strategy import Base_Exploration_Strategy

class OU_Noise_Exploration(Base_Exploration_Strategy):
    '''
        https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/exploration_strategies/OU_Noise_Exploration.py
    '''

    """Ornstein-Uhlenbeck noise process exploration strategy"""
    def __init__(self, action_size, config):
        super(OU_Noise_Exploration, self).__init__(config)
        self.noise = OUNoise(action_size, self.config['noise_scale'], self.config['noise_mu'], self.config['noise_theta'], self.config['noise_sigma'])

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action = action_info["action"]
        action += self.noise.noise()
        return action

    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        raise ValueError("Must be implemented")

    def reset(self):
        """Resets the noise process"""
        self.noise.reset()