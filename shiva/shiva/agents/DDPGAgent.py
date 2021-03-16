import numpy as np
import random
import torch
import copy
from torch.distributions import Categorical
from torch.nn.functional import softmax
from shiva.helpers.calc_helper import two_point_formula
from shiva.agents.Agent import Agent
from shiva.helpers.utils import Noise as noise
from shiva.helpers.networks_helper import mod_optimizer
from shiva.networks.DynamicLinearNetwork import DynamicLinearNetwork, SoftMaxHeadDynamicLinearNetwork


class DDPGAgent(Agent):
    """ DDPG Agent Class

    Used for getting discrete and continuous actions. Also has saving and loading functions.
    Also has functions to modify and evolve hyper-parameters.

    Args:
        id (int): Id to identify the agent.
        obs_space(int): Number of observations for the agent to expect.
        acs_space(dict): Number of actions the agent is expected to produce seperate by type of action
            space.
        configs: Hyperparameters used to set up the agent.
    Returns:
        None.
    """
    def __init__(self, id:int, obs_space:int, acs_space:dict, configs):
        super(DDPGAgent, self).__init__(id, obs_space, acs_space, configs)
        torch.manual_seed(self.manual_seed)
        np.random.seed(self.manual_seed)
        self.log(f"MANUAL SEED {self.manual_seed}", verbose_level=2)

        self.discrete = acs_space['discrete']
        self.continuous = acs_space['continuous']
        self.param = acs_space['discrete']

        if self.continuous == 0:
            self.action_space = 'discrete'
            self.actor_input = obs_space
            self.actor_output = self.discrete
            self.get_action = self.get_discrete_action
        elif self.discrete == 0:
            self.action_space = 'continuous'
            self.actor_input = obs_space
            self.actor_output = self.continuous
            self.get_action = self.get_continuous_action
            if not hasattr(self, 'actions_range'):
                self.actions_range = [-1, 1]
        else:
            raise NotImplemented
            # self.action_space = 'parametrized'
            # self.actor_input = obs_space
            # self.actor_output = self.discrete + self.param
            # self.get_action = self.get_parameterized_action

        self.instantiate_networks()

    def instantiate_networks(self) -> None:
        """ Creates the networks that agent will use for inference or evaluation.

        Uses the network section of the config along with acs_space and obs_space to
        create the networks. Also creates the target networks.

        Returns:
            None
        """
        self.hps = ['actor_learning_rate', 'critic_learning_rate']
        self.hps += ['epsilon', 'noise_scale']
        self.hps += ['epsilon_start', 'epsilon_end', 'epsilon_episodes', 'noise_start', 'noise_end', 'noise_episodes']

        self.ou_noise = noise.OUNoiseTorch(sum(self.actor_output), self.noise_scale)
        self.hp_random = self.hp_random if hasattr(self, 'hp_random') else False

        if self.hp_random:
            self.epsilon = np.random.uniform(self.epsilon_range[0], self.epsilon_range[1])
            self.noise_scale = np.random.uniform(self.ou_range[0], self.ou_range[1])
            self.actor_learning_rate = np.random.uniform(self.agent_config['lr_uniform'][0], self.agent_config['lr_uniform'][1]) / np.random.choice(self.agent_config['lr_factors'])
            self.critic_learning_rate = np.random.uniform(self.agent_config['lr_uniform'][0], self.agent_config['lr_uniform'][1]) / np.random.choice(self.agent_config['lr_factors'])
        else:
            self.epsilon = self.epsilon_start
            self.noise_scale = self.noise_start
            self.actor_learning_rate = self.agent_config['actor_learning_rate']
            self.critic_learning_rate = self.agent_config['critic_learning_rate']

        if 'MADDPG' not in str(self):
            self.critic_input_size = self.actor_input + sum(self.actor_output)

        self.net_names = ['actor', 'target_actor', 'critic', 'target_critic', 'actor_optimizer', 'critic_optimizer']

        if self.action_space == 'continuous':
            self.actor = DynamicLinearNetwork(self.actor_input, sum(self.actor_output), self.networks_config['actor'])
            self.exploration_policy = torch.distributions.uniform.Uniform(low=self.actions_range[0], high=self.actions_range[1])
        elif self.action_space == 'discrete':
            self.actor = SoftMaxHeadDynamicLinearNetwork(self.actor_input, self.actor_output, self.actor_output, self.networks_config['actor'])
            self.exploration_policy = torch.distributions.uniform.Uniform(low=0, high=1)

        self.target_actor = copy.deepcopy(self.actor)
        '''If want to save memory on an MADDPG (not multicritic) run, put critic networks inside if statement'''
        self.critic = DynamicLinearNetwork(self.critic_input_size, 1, self.networks_config['critic'])
        self.target_critic = copy.deepcopy(self.critic)
        self.actor_optimizer = self.optimizer_function(params=self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = self.optimizer_function(params=self.critic.parameters(), lr=self.critic_learning_rate)

    # It's now in the Abstract Agent
    # def to_device(self, device):
    #     self.device = device
    #     self.actor.to(self.device)
    #     self.target_actor.to(self.device)
    #     self.critic.to(self.device)
    #     self.target_critic.to(self.device)

    def get_discrete_action(self, observation, acs_mask, step_count, evaluate=False, one_hot=False, *args, **kwargs):
        """ Produces a discrete action.

        The action could be either an inference or evaluation action.

        Args:
             observation: State the environment is in.
             step_count: How many total steps the agent has seen.
             evaluate: Whether or not you want a noise free deterministic action.
             one_hot: Whether you want a one hot encoded action or raw.

        Returns:
            A list containing probabilities for each action possible in the step.
        """
        # print(str(self), observation, acs_mask)

        observation = torch.tensor(observation).to(self.device).float()
        if len(observation.shape)>1:
            self._output_dimension = (*observation.shape[:-1], sum(self.actor_output))
        else:
            self._output_dimension = (sum(self.actor_output),)
        self.ou_noise.set_output_dim(self._output_dimension)

        if evaluate:
            action = self.actor(observation).detach().cpu()
        else:
            if self.is_exploring() or self.is_e_greedy():
                action = self.exploration_policy.sample(torch.Size([*self._output_dimension]))
                _action_debug = "Random: {}".format(action)
            else:
                # Forward pass
                action = self.actor(observation).detach().cpu() + self.ou_noise.noise()
                # Mask actions
                action[acs_mask] = 0
                # Normalize each individual branch
                _cum_ix = 0
                for ac_dim in self.actor_output:
                    _branch_action = torch.abs(action[_cum_ix:ac_dim+_cum_ix])
                    action[_cum_ix:ac_dim+_cum_ix] = _branch_action / _branch_action.sum()
                    _cum_ix += ac_dim
                _action_debug = "Net: {}".format(action)

            self.log(f"Obs {observation.shape} Acs {action.shape}\nObs {observation} Acs {_action_debug}", verbose_level=3)
        # if one_hot:
        #     action = action2one_hot(action.shape[0], torch.argmax(action).item())

        # until this point the action was a tensor, we are returning a python list - needs to be checked.
        return action.tolist()

    def get_continuous_action(self, observation, acs_mask, step_count, evaluate=False, *args, **kwargs):
        """ Produces a continuous action.

        The action could be either an inference or evaluation action.

        Args:
             observation: State the environment is in.
             step_count: How many total steps the agent has seen.
             evaluate: Whether or not you want a noise free deterministic action.

        Returns:
            A list containing probabilities for each action possible in the step.
        """
        observation = torch.tensor(observation).to(self.device).float()
        if len(observation.shape)>1:
            self._output_dimension = (*observation.shape[:-1], sum(self.actor_output))
        else:
            self._output_dimension = (sum(self.actor_output),)
        self.ou_noise.set_output_dim(self._output_dimension)

        if evaluate:
            action = self.actor(observation).detach()
        else:
            if self.is_exploring() or self.is_e_greedy():
                action = self.exploration_policy.sample(torch.Size([*self._output_dimension]))
                self.log(f"** Random action {action.tolist()}", verbose_level=1)
            else:
                action = self.actor(observation).detach().cpu() + self.ou_noise.noise()
                action = torch.clamp(action, min=self.actions_range[0], max=self.actions_range[1])
                self.log(f"Network action {action.tolist()}", verbose_level=1)
        return action.tolist()

    def get_parameterized_action(self, observation, step_count, evaluate=False) -> None:
        """ Produces a parameterized action.

        Currently not implemented. The action could be either an inference or evaluation action.

        Args:
             observation: State the environment is in.
             step_count: How many total steps the agent has seen.
             evaluate: Whether or not you want a noise free action.

        Returns:
            A list containing probabilities for each action possible in the step.
        """
        raise NotImplemented
        # if self.evaluate:
        #     observation = torch.tensor(observation).to(self.device)
        #     action = self.actor(observation.float())
        # else:
        #     # if step_count > self.exploration_steps:
        #     # else:
        #     observation = torch.tensor(observation).to(self.device)
        #     action = self.actor(observation.float())
        # # return action.tolist()
        # return action[0]

    def get_imitation_action(self, observation: np.ndarray) -> np.ndarray:
        """ Produces an imitated action.

        Args:
             observation: State the environment is in.

        Returns:
            A numpy array containing probabilities for each action possible in the step.
        """
        observation = torch.tensor(observation).to(self.device)
        action = self.actor(observation.float())
        return action[0]

    # def save(self, save_path, step):
    #     torch.save(self.actor, save_path + '/actor.pth')
    #     torch.save(self.target_actor, save_path + '/target_actor.pth')
    #     torch.save(self.critic, save_path + '/critic.pth')
    #     torch.save(self.target_critic, save_path + '/target_critic.pth')
    #     torch.save(self.actor_optimizer, save_path + '/actor_optimizer.pth')
    #     torch.save(self.critic_optimizer, save_path + '/critic_optimizer.pth')

    def copy_hyperparameters(self, evo_agent) -> None:
        """ Copies the hyperparameters of a passed in agent.

        Args:
            evo_agent(DDPGAgent): An evolved agent from which to copy hyperparameters from.

        Returns:
            None
        """
        self.actor_learning_rate = evo_agent.actor_learning_rate
        self.critic_learning_rate = evo_agent.critic_learning_rate
        self._update_optimizers()
        self.epsilon = evo_agent.epsilon
        self.noise_scale = evo_agent.noise_scale

    def copy_weights(self, evo_agent) -> None:
        """ Used to copy this evolved agent's weights into this agent.

        Args:
            evo_agent (DDPGAgent): An evolved agent from which to copy hyperparameters from.
        Returns:
            None
        """
        self.num_evolutions['truncate'] += 1
        self.actor.load_state_dict(evo_agent.actor.to(self.device).state_dict())
        self.target_actor.load_state_dict(evo_agent.target_actor.to(self.device).state_dict())
        self.critic.load_state_dict(evo_agent.critic.to(self.device).state_dict())
        self.target_critic.load_state_dict(evo_agent.target_critic.to(self.device).state_dict())
        self.actor_optimizer.load_state_dict(evo_agent.actor_optimizer.state_dict())
        self.critic_optimizer.load_state_dict(evo_agent.critic_optimizer.state_dict())

    def perturb_hyperparameters(self, perturb_factor) -> None:
        """ Slight mutates this agent's weights by a factor.
        Args:
            perturb_factor(float): Factor by which to modify agent weights.
        Returns:
            None
        """
        self.num_evolutions['perturb'] += 1
        self.actor_learning_rate *= perturb_factor
        self.critic_learning_rate *= perturb_factor
        self._update_optimizers()
        self.epsilon *= perturb_factor
        self.noise_scale *= perturb_factor

    def resample_hyperparameters(self) -> None:
        """ Gets new weights for this agent from a set predetermined intervals in the configs.

        Returns:
            None
        """
        self.num_evolutions['resample'] += 1
        self.actor_learning_rate = np.random.uniform(self.agent_config['lr_uniform'][0], self.agent_config['lr_uniform'][1]) / np.random.choice(self.agent_config['lr_factors'])
        self.critic_learning_rate = np.random.uniform(self.agent_config['lr_uniform'][0], self.agent_config['lr_uniform'][1]) / np.random.choice(self.agent_config['lr_factors'])
        self._update_optimizers()
        self.epsilon = np.random.uniform(self.epsilon_range[0], self.epsilon_range[1])
        self.noise_scale = np.random.uniform(self.ou_range[0], self.ou_range[1])

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
        self.actor_learning_rate *= self.lr_decay['factor']
        if 'MADDPG' not in str(self):
            self.critic_learning_rate *= self.lr_decay['factor']
            self.log(f"Decay Actor and Critic LR: {self.actor_learning_rate} {self.critic_learning_rate}", verbose_level=3)
        else:
            self.log(f"Decay Actor LR: {self.actor_learning_rate}", verbose_level=3)
        self._update_optimizers()

    def restore_learning_rate(self) -> None:
        """ Increases the networks' learning rate.

        Returns:
            None
        """
        if self.actor_learning_rate < self.configs['Agent']['actor_learning_rate']:
            self.actor_learning_rate /= self.lr_decay['factor']
            self.log(f"Increment Actor LR {self.actor_learning_rate}", verbose_level=3)
        else:
            self.actor_learning_rate = self.configs['Agent']['actor_learning_rate']
            self.log(f"Actor LR Restored {self.actor_learning_rate}", verbose_level=3)

        if 'MADDPG' not in str(self):
            if self.critic_learning_rate < self.configs['Agent']['critic_learning_rate']:
                self.critic_learning_rate /= self.lr_decay['factor']
                self.log(f"Increment Critic LR {self.actor_learning_rate}", verbose_level=3)
            else:
                self.critic_learning_rate = self.configs['Agent']['critic_learning_rate']
            self.log(f"Critic LR Restored {self.actor_learning_rate}", verbose_level=3)

        self._update_optimizers()

    def _update_optimizers(self):
        self.actor_optimizer = mod_optimizer(self.actor_optimizer, {'lr': self.actor_learning_rate})
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
        est_exploration_episodes = self.average_exploration_episodes_performed() if not hasattr(self, 'exploration_episodes') else self.exploration_episodes
        new_scale = two_point_formula(self.done_count, (est_exploration_episodes, self.epsilon_start), (est_exploration_episodes+self.epsilon_episodes, self.epsilon_end))
        new_scale = min(new_scale, self.epsilon_start)
        new_scale = max(new_scale, self.epsilon_end)
        return new_scale

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
        est_exploration_episodes = self.average_exploration_episodes_performed() if not hasattr(self, 'exploration_episodes') else self.exploration_episodes
        new_scale = two_point_formula(self.done_count, (est_exploration_episodes, self.noise_start), (est_exploration_episodes+self.noise_episodes, self.noise_end))
        new_scale = min(new_scale, self.noise_start)
        new_scale = max(new_scale, self.noise_end)
        return new_scale

    def average_exploration_episodes_performed(self):
        return self.exploration_steps / self.average_episode_length if self.average_episode_length != 0 else 0

    @property
    def average_episode_length(self):
        return self.step_count / self.done_count if self.done_count != 0 else 0

    def reset_noise(self) -> None:
        """ Resets the Agent's OU Noise

        This is super important otherwise the noise will compound.

        Returns:
            None
        """
        self.ou_noise.reset()

    def get_metrics(self):
        """Gets the metrics so they can be passed to tensorboard.

        Returns:
             A tuple of metrics.
        """
        return [
            ('{}/Actor_Learning_Rate'.format(self.role), self.actor_learning_rate),
            ('{}/Critic_Learning_Rate'.format(self.role), self.critic_learning_rate),
            ('{}/Epsilon'.format(self.role), self.epsilon),
            ('{}/Noise_Scale'.format(self.role), self.noise_scale),
        ]

    def get_module_and_classname(self):
        """ Returns the name of the module and class name.

        Returns:
            A string representation of the module and class.

        """
        return ('shiva.agents', 'DDPGAgent.DDPGAgent')

    def __str__(self):
        return f"<DDPGAgent(id={self.id}, role={self.role}, S/E/U={self.step_count}/{self.done_count}/{self.num_updates}, T/P/R={self.num_evolutions['truncate']}/{self.num_evolutions['perturb']}/{self.num_evolutions['resample']} noise/epsilon={round(self.noise_scale, 2)}/{round(self.epsilon, 2)} device={self.device})>"