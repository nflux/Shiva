import copy
import torch.optim
import numpy as np
import random
import shiva.helpers.misc as misc
from shiva.helpers.networks_helper import mod_optimizer
from typing import List
from shiva.helpers.utils import Noise as noise
from shiva.agents.Agent import Agent
import shiva.helpers.networks_handler as nh
from shiva.helpers.misc import action2one_hot


class DQNAgent(Agent):
    def __init__(self, id: int, obs_space: int, acs_space: int, configs: dict):
        super(DQNAgent, self).__init__(id, obs_space, acs_space, configs)
        self.id = id
        self.action_space = acs_space
        self.critic_input = obs_space + self.action_space
        self.critic_output = 1
        self.agent_config = configs['Agent']
        self.net_config = self.configs['Network']

        if isinstance(self.agent_config['critic_learning_rate'], list):
            self.learning_rate = random.uniform(self.agent_config['learning_rate'][0], self.agent_config['learning_rate'][1])
        else:
            self.learning_rate = self.agent_config['critic_learning_rate']

        self.instantiate_network()

    def instantiate_network(self) -> None:
        """ Creates the crtic network the DQNAgent will use for inference or evaluation.

        Uses the network section of the config along with acs_space and obs_space to
        create the network. Also creates the target network.

        Returns:
            None
        """
        self.hps = ['critic_learning_rate']
        self.hps += ['epsilon', 'noise_scale']
        self.hps += ['epsilon_start', 'epsilon_end', 'epsilon_episodes', 'noise_start', 'noise_end', 'noise_episodes']

        self.hp_random = self.hp_random if hasattr(self, 'hp_random') else False

        if self.hp_random:
            self.epsilon = np.random.uniform(self.epsilon_range[0], self.epsilon_range[1])
            self.noise_scale = np.random.uniform(self.ou_range[0], self.ou_range[1])
            self.critic_learning_rate = np.random.uniform(self.agent_config['lr_uniform'][0], self.agent_config['lr_uniform'][1]) / np.random.choice(self.agent_config['lr_factors'])
        else:
            self.epsilon = self.epsilon_start
            self.noise_scale = self.noise_start
            self.critic_learning_rate = self.agent_config['critic_learning_rate']

        self.ou_noise = noise.OUNoiseTorch(self.action_space, self.noise_scale)

        self.net_names = ['critic', 'target_critic', 'critic_optimizer']
        self.exploration_policy = torch.distributions.uniform.Uniform(low=0, high=1)

        '''If want to save memory on an MADDPG (not multicritic) run, put critic networks inside if statement'''
        self.critic = nh.DynamicLinearSequential(
                                        self.critic_input,
                                        self.critic_output,
                                        self.net_config['network']['layers'],
                                        nh.parse_functions(torch.nn, self.net_config['network']['activation_function']),
                                        self.net_config['network']['last_layer'],
                                        self.net_config['network']['output_function']
        )
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = getattr(torch.optim, self.agent_config['optimizer_function'])(params=self.critic.parameters(), lr=self.critic_learning_rate)
        # self.critic_optimizer = self.optimizer_function(params=self.critic.parameters(), lr=self.critic_learning_rate)

    def get_action(self, obs, step_n=0, evaluate=False):
        """
            This method iterates over all the possible actions to find the one with the highest Q value
        """
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs).float()
        if evaluate:
            return self.find_best_action(self.critic, obs)
        # if step_n < self.exploration_steps or random.uniform(0, 1) < max(self.epsilon_end, self.epsilon_start - (step_n * self.epsilon_decay)):
        if step_n < self.exploration_steps or self.is_e_greedy(step_n):
            '''Random action or e-greedy exploration'''
            # check if obs is a batch!
            if len(obs.shape) > 1:
                # it's a batch operation!
                action = [self.get_random_action() for _ in range(obs.shape[0])]
                # print('random batch action')
            else:
                action = self.get_random_action()
                # print("random act")
        else:
            # print('action! {} {}'.format(obs, obs.shape))
            action = self.find_best_action(self.critic, obs)
        # print("From DQAgent Acs {}".format(action))
        return action

    def get_action_target(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs).float()
        return self.find_best_action(self.target_critic, obs)

    def get_random_action(self):
        action_idx = random.sample(range(self.acs_space), 1)[0]
        action = list(action2one_hot(self.acs_space, action_idx))
        return action

    def find_best_action(self, network, observation) -> List:
        if len(observation.shape) > 1:
            '''This is for batch operation, as we need to find the highest Q for each obs'''
            return [self.find_best_action_from_tensor(network, ob) for ob in observation]
        else:
            return self.find_best_action_from_tensor(network, observation)

    def find_best_action_from_tensor(self, network, obs_v) -> List:
        """
            Iterates over the action space to find the action with the highest Q value

            Input
                network         policy network to be used
                observation     observation from the environment

            Returns
                A one-hot encoded list
        """
        obs_v = obs_v.to(self.device)
        best_q, best_act_v = float('-inf'), torch.zeros(self.action_space).to(self.device)
        for i in range(self.action_space):
            act_v = misc.action2one_hot_v(self.action_space, i).to(self.device)
            q_val = network(torch.cat([obs_v.float(), act_v.float()], dim=-1))
            if q_val > best_q:
                best_q = q_val
                best_act_v = act_v
        best_act = best_act_v.tolist()
        return best_act

    def find_best_imitation_action(self, observation: np.ndarray) -> np.ndarray:
        obs_v = torch.tensor(observation).float().to(self.device)
        best_q, best_act_v = float('-inf'), torch.zeros(self.acs_space).to(self.device)
        for i in range(self.acs_space):
            act_v = misc.action2one_hot_v(self.acs_space,i)
            q_val = self.critic(torch.cat([obs_v, act_v.to(self.device)]))
            if q_val > best_q:
                best_q = q_val
                best_act_v = act_v
        best_act = best_act_v
        return np.argmax(best_act).numpy()

    def save(self, save_path, step):
        torch.save(self.critic, save_path + '/critic.pth')
        torch.save(self.target_critic, save_path + '/target_critic.pth')

    def copy_weights(self, evo_agent):
        print('Copying Weights')
        self.critic_learning_rate = evo_agent.critic_learning_rate
        self.critic_optimizer = copy.deepcopy(evo_agent.critic_optimizer)
        self.critic = copy.deepcopy(evo_agent.critic)
        self.target_critic = copy.deepcopy(evo_agent.target_critic)
        print('Copied Weights')

    def perturb_hyperparameters(self,perturb_factor):
        print('Pertubing HP')
        self.critic_learning_rate = self.learning_rate * perturb_factor
        self.critic_optimizer = self.optimizer_function(params=self.policy.parameters(), lr=self.learning_rate)
        print('Pertubed HP')

    def resample_hyperparameters(self):
        print('Resampling')
        self.critic_learning_rate = random.uniform(self.agent_config['learning_rate'][0],self.agent_config['learning_rate'][1])
        self.critic_optimizer = self.optimizer_function(params=self.policy.parameters(), lr=self.learning_rate)
        print('Resampled')

    def recalculate_hyperparameters(self, done_count=None) -> None:
        """ Updates the epsilon and noise hyperparameters over time.

        Args:
            done_count (int): Episodic independent value controlling the hyperparameters.
        Returns:
            None
        """
        if done_count is None:
            done_count = self.done_count if self.done_count != 0 else 1
        self.update_epsilon_scale(done_count)
        self.update_noise_scale(done_count)

    def decay_learning_rate(self) -> None:
        """ Increases the critics' learning rate.

        Returns:
            None
        """
        self.critic_learning_rate *= self.lr_decay['factor']
        self.log(f"Decay Critic LR: {self.critic_learning_rate}", verbose_level=3)

        self._update_optimizers()

    def restore_learning_rate(self) -> None:
        """ Increases the networks' learning rate.

        Returns:
            None
        """
        if self.critic_learning_rate < self.configs['Agent']['critic_learning_rate']:
            self.critic_learning_rate /= self.lr_decay['factor']
            self.log(f"Increment Critic LR {self.critic_learning_rate}", verbose_level=3)
        else:
            self.critic_learning_rate = self.configs['Agent']['critic_learning_rate']

        self._update_optimizers()

    def _update_optimizers(self):
        self.critic_optimizer = mod_optimizer(self.critic_optimizer, {'lr': self.critic_learning_rate})

    def update_epsilon_scale(self, done_count=None) -> None:
        """To be called by the Learner before saving
        Decreases the epsilon scale.
        Args:
            done_count (int):
        Returns:
            None
        """
        if self.hp_random is False:
            if done_count is None:
                done_count = self.done_count
            # if self.is_exploring():
            #     return self.epsilon_start
            self.epsilon = self._get_epsilon_scale(done_count)

    def _get_epsilon_scale(self, done_count=None):
        if done_count is None:
            done_count = self.done_count
        return max(self.epsilon_end, self.decay_value(self.epsilon_start, self.epsilon_episodes, (done_count - self._average_exploration_episodes_performed()), degree=self.epsilon_decay_degree))

    def update_noise_scale(self, done_count=None) -> None:
        """To be called by the Learner before saving
        Decreases the noise scale.
        Args:
            done_count (int):
        Returns:
            None
        """
        if self.hp_random is False:
            if done_count is None:
                done_count = self.done_count
            self.noise_scale = self._get_noise_scale(done_count)
            self.ou_noise.set_scale(self.noise_scale)

    def _get_noise_scale(self, done_count=None):
        if done_count is None:
            done_count = self.done_count
        return max(self.noise_end, self.decay_value(self.noise_start, self.noise_episodes, (done_count - (self._average_exploration_episodes_performed())), degree=self.noise_decay_degree))

    def _average_exploration_episodes_performed(self):
        return self.exploration_steps / (self.step_count / self.done_count) if (self.step_count != 0 and self.done_count != 0) else 0

    def reset_noise(self) -> None:
        """ Resets the Agent's OU Noise

        This is super important otherwise the noise will compound.

        Returns:
            None
        """
        self.ou_noise.reset()

    def decay_value(self, start, decay_end_step, current_step_count, degree=1) -> float:
        """ Decays a value from an initial to final value.

        Can be linear, or polynomial.

        Returns:
            A new float value.
        """
        return start - start * ((current_step_count / decay_end_step) ** degree)

    def is_e_greedy(self, step_count=None) -> bool:
        """ Checks if an action should be random or inference.

        Args:
            step_count (int): Episodic counter that controls the epsilon.

        Returns:
            Boolean indicating whether or not to take a random action.
        """
        if step_count is None:
            step_count = self.step_count
        if step_count > self.exploration_steps:
            step_count = step_count - self.exploration_steps # don't count explorations steps
            return random.uniform(0, 1) < self.epsilon
        else:
            return True

    def is_exploring(self, current_step_count=None) -> bool:
        """ Checks if an action should be random or inference.

        Args:
            step_count (int): Episodic counter that controls the noise.

        Returns:
            Boolean indicating whether or not to add noise to an action.
        """
        if hasattr(self, 'exploration_episodes'):
            if current_step_count is None:
                current_step_count = self.done_count
            _threshold = self.exploration_episodes
        else:
            if current_step_count is None:
                current_step_count = self.step_count
            _threshold = self.exploration_steps
        return current_step_count < _threshold

    def get_metrics(self):
        """Gets the metrics so they can be passed to tensorboard.

        Returns:
             A tuple of metrics.
        """
        return [
            ('{}/Critic_Learning_Rate'.format(self.role), self.critic_learning_rate),
            ('{}/Epsilon'.format(self.role), self.epsilon),
            ('{}/Noise_Scale'.format(self.role), self.noise_scale),
        ]

    def get_module_and_classname(self):
        return ('shiva.agents', 'DQNAgent.DQNAgent')

    def __str__(self):
        return '<DQNAgent(id={}, steps={}, episodes={})>'.format(self.id, self.step_count, self.done_count)
