import numpy as np
import gym
import torch

from shiva.envs.Environment import Environment

import sigma_graph


class MultiAgentGraphEnv(Environment):
    """ Gym Wrapper for interfacing with SigmaGraphEnvs, a set of custom multi-agent OpenAI Gym Environment scenarios.
        Env package is inside of 'sigma_graph'
    """

    def __init__(self, configs, **kwargs):
        super(MultiAgentGraphEnv, self).__init__(configs)

        ''' Start Gym MA environment '''
        self.gym_configs = self.passing_configs(configs, **kwargs)
        self.env = gym.make(self.env_name, **self.gym_configs)
        self.env.reset()

        ''' Set initial values '''
        self.roles = [role for role in self.env.learning_agent]
        self.num_agents = len(self.roles)

        self.num_instances_per_role = {role: 1 for role in self.roles}
        self.num_instances_per_env = 1

        self.observation_space = {role: self.get_observation_space_from_env(self.env.observation_space[ix]) for ix, role in enumerate(self.roles)}
        self.action_space = {role: self.get_action_space_from_env(self.env.action_space[ix]) for ix, role in enumerate(self.roles)}
        self.current_acs_mask = {role: [] for role in self.roles}
        self.prev_acs_mask = {role: [] for role in self.roles}

        '''Init session cumulative metrics'''
        self.actions = {role: [] for role in self.roles}
        self.observations = {role: [] for role in self.roles}
        self.rewards = {role: [] for role in self.roles}
        self.dones = {role: [] for role in self.roles}

        self.steps_per_episode = 0
        self.reward_per_step = {}
        self.reward_per_episode = {}
        self.reward_total = {}

        self.step_count = 0
        self.done_count = 0

        '''Reset Metrics'''
        self.reset()

    def reset(self, force=True, **kwargs):
        obs = self.env.reset()
        self.observations = {role: obs[ix] for ix, role in enumerate(self.roles)}
        self.steps_per_episode = 0
        self.reward_per_step = {role: 0 for role in self.roles}
        self.reward_per_episode = {role: 0 for role in self.roles}
        self.step_count = 0
        if force:
            self.reward_total = {role: 0 for role in self.roles}
            self.done_count = 0

    def step(self, _actions):
        # generate list of actions for each agent and send to env
        # if not torch.is_tensor(_actions):
        #     _actions = torch.tensor(_actions)
        action4gym = []
        for idx, role in enumerate(self.roles):
            # process Discrete and MultiDiscrete action space
            if self.action_space[role]['continuous'] == 0:
                _act_dims = len(self.action_space[role]['acs_space'])
                if _act_dims == 1:
                    action4gym.append(self.action_selection(_actions[idx]))
                # process branched actions
                elif _act_dims > 1:
                    # set start and end indexing boundaries for each action branch
                    _act_dims = len(self.action_space[role]['acs_space'])
                    _bound_e = [0] * _act_dims
                    for _ in range(_act_dims):
                        _bound_e[_] = sum(self.action_space[role]['acs_space'][:_ + 1])
                    _bound_s = [0] + _bound_e[:-1]
                    # concatenate all branched actions per agent
                    acts_prob = _actions[idx]
                    agent_acts = []
                    for _ in range(_act_dims):
                        act_prob = acts_prob[_bound_s[_]:_bound_e[_]]
                        agent_acts.append(self.action_selection(act_prob))
                    self.log("action for agent:{} raw:{} 4gym:{}".format(role, acts_prob, agent_acts), verbose_level=4)
                    action4gym.append(agent_acts)
                else:
                    assert "Unexpected action space shape:{}".format(self.action_space[role]['acs_space'])
            else:
                assert "Do not support action space:{}".format(self.action_space[role])
            self.actions[role] = action4gym[idx]
        self.log("actions4gym:{}".format(action4gym), verbose_level=4)
        # run step function inside env scenario, update states based on given actions
        obs, rewards, dones, _ = self.env.step(action4gym)

        self.prev_acs_mask = self.current_acs_mask.copy()
        self.current_acs_mask = self.get_current_action_masking()

        self.observations = {role: obs[idx] for idx, role in enumerate(self.roles)}
        self.rewards = {role: rewards[idx] for idx, role in enumerate(self.roles)}
        self.dones = {role: dones[idx] for idx, role in enumerate(self.roles)}
        self.steps_per_episode += self.num_instances_per_env
        self.step_count += self.num_instances_per_env
        self.done_count += int(self.dones[self.roles[0]])

        for role in self.roles:
            self.reward_per_step[role] += self.rewards[role]
            self.reward_per_episode[role] += self.rewards[role]
            self.reward_total[role] += self.reward_per_episode[role]

        return self.observations, self.rewards, self.dones, {'action': self.actions}

    def is_done(self):
        return self.steps_per_episode >= self.episode_max_length

    def get_current_action_masking(self, per_role=None):
        if "invalid_masked" in self.env.configs and self.env.configs["invalid_masked"] is True:
            _action_mask = self.env.action_mask
        else:
            _action_mask = self.get_empty_masks()

        if per_role is not None and per_role in self.roles:
            idx = self.roles.index(per_role)
            action_mask = _action_mask[idx]
        else:
            action_mask = {role: _action_mask[idx] for idx, role in enumerate(self.roles)}
        return action_mask

    def get_empty_masks(self):
        empty_mask = {role: np.zeros(sum(self.action_space[role]['acs_space']), dtype=np.bool_) for role in self.roles}
        return empty_mask

    def get_action_space_from_env(self, agent_action_space):
        if isinstance(agent_action_space, gym.spaces.Discrete):
            return {
                'discrete': (agent_action_space.n,),
                'continuous': 0,
                'param': 0,
                'acs_space': (agent_action_space.n,),
                'actions_range': []
            }
        elif isinstance(agent_action_space, gym.spaces.MultiDiscrete):
            return {
                'discrete': tuple(agent_action_space.nvec),
                'continuous': 0,
                'param': 0,
                'acs_space': tuple(agent_action_space.nvec),
                'actions_range': []
            }
        else:
            assert "[GymEnv] Only support Discrete and MultiDiscrete action spaces"

    def get_observation_space_from_env(self, agent_obs_space) -> int:  # unfold gym.space
        observation_space = 1
        if agent_obs_space.shape != ():
            for i in range(len(agent_obs_space.shape)):
                observation_space *= agent_obs_space.shape[i]
        else:
            observation_space = agent_obs_space.n
        assert observation_space > 1, "Error processing Obs space? got {}".format(agent_obs_space)
        return observation_space

    def get_actions(self):
        return list(self.actions.values())

    def get_observations(self):
        return list(self.observations.values())

    def get_rewards(self):
        return list(self.rewards.values())

    def get_action(self, role):
        return list(self.actions[role])

    def get_observation(self, role):
        return list(self.observations[role])

    def get_reward(self, role):
        return list(self.rewards[role])

    def get_total_reward(self):
        return list(self.reward_total.values())

    def get_reward_episode(self, roles=True):
        return self.reward_per_episode

    def get_metrics(self, episodic=True):
        """Returns the metrics of the environment."""
        metrics = {role: self.get_role_metrics(role, episodic) for role in self.roles}
        return list(metrics.values())

    def get_role_metrics(self, role=None, episodic=True):
        if not episodic:
            metrics = [
                ('Reward/Per_Step', self.reward_per_step[role])
            ]
        else:
            metrics = [
                ('Reward/Per_Episode', self.reward_per_episode[role]),
                ('Agent/Steps_Per_Episode', self.steps_per_episode)
            ]
        return metrics

    def passing_configs(self, config, **kwargs):
        # TODO>>> define config args; check env_name & configs
        env_configs = self.configs['Environment']["gym_config"] if "gym_config" in self.configs['Environment'] else {}
        if "env_path" not in env_configs:
            env_configs["env_path"] = "../../squad/"
        if "max_step" not in env_configs:
            env_configs["max_step"] = self.episode_max_length
        self.log("passing configs:{}".format(env_configs), verbose_level=4)
        return env_configs

    def load_viewer(self):
        return False

    def close(self):
        self.env.close()
        delattr(self, 'env')
