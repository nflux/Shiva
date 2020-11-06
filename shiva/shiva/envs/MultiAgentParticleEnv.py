import numpy as np
import gym

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

from shiva.envs.Environment import Environment

class MultiAgentParticleEnv(Environment):
    _GYM_VERSION_ = '0.10.5'
    def __init__(self, config):
        assert gym.__version__ == self._GYM_VERSION_, "MultiAgentParticleEnv requires Gym {}. Run 'pip uninstall gym' and 'pip install gym==0.10.5' ".format(self._GYM_VERSION_)
        super(MultiAgentParticleEnv, self).__init__(config)
        self._connect()
        self.set_initial_values()

    def _connect(self):
        # load scenario from script
        _load_name = self.env_name
        if '.py' not in _load_name:
            _load_name += '.py'
        self.scenario = scenarios.load(_load_name).Scenario()
        # create world
        self.world = self.scenario.make_world()
        # create multiagent environment
        self.env = MultiAgentEnv(self.world, self.scenario.reset_world, self.scenario.reward, self.scenario.observation, info_callback=None, shared_viewer=self.share_viewer)
        # render call to create viewer window (necessary only for interactive policies)
        self.display()
        self.step_data = self.env.reset()

    def set_initial_values(self):
        '''Unique Agent Behaviours'''
        self.num_instances_per_env = 1
        self.num_agents = self.env.n
        self.roles = [a.name for a in self.env.agents]

        self.action_space = {role:self.get_action_space_from_env(self.env.action_space[ix]) for ix, role in enumerate(self.roles)}
        self.observation_space = {role:self.get_observation_space_from_env(self.env.observation_space[ix]) for ix, role in enumerate(self.roles)}

        self.num_instances_per_role = {role:1 for role in self.roles}
        self.num_instances_per_env = 1

        '''Init session cumulative metrics'''
        self.reward_total = {role:0 for role in self.roles}
        self.step_count = 0
        self.done_count = 0

        # reward function
        self.rewards_wrap = lambda x: x
        if hasattr(self, 'normalize_reward') and self.normalize:
            self.rewards_wrap = self.normalize_reward

        '''Reset Metrics'''
        self.reset()

    def reset(self, *args, **kwargs):
        '''
            To be called by Shiva Learner
            It's just to reinitialize our metrics. Unity resets the environment on its own.
        '''
        self.steps_per_episode = 0
        self.temp_done_counter = 0
        self.reward_per_step = {role:0 for role in self.roles}
        self.reward_per_episode = {role:0 for role in self.roles}

        obs = self.env.reset()
        self.observations = {role:obs[ix] for ix, role in enumerate(self.roles)}

    def step(self, actions):
        self.actions = {role:self._clean_actions(role, actions[ix]) for ix, role in enumerate(self.roles)}
        obs, rew, don, _ = self.env.step(list(self.actions.values()))
        self.observations = {role:obs[ix] for ix, role in enumerate(self.roles)}
        self.rewards = {role:self.rewards_wrap(rew[ix]) for ix, role in enumerate(self.roles)}

        # maybe overwrite the done - not sure if env tells when is Done
        self.dones = {role:don[ix] for ix, role in enumerate(self.roles)}

        '''
            Metrics collection
                Episodic # of steps             self.steps_per_episode --> is equal to the amount of instances on Unity, 1 Shiva step could be a couple of Unity steps
                Cumulative # of steps           self.step_count
                Temporary episode count         self.temp_done_counter --> used for the is_done() call. Zeroed on reset().
                Cumulative # of episodes        self.done_count
                Step Reward                     self.reward_per_step
                Episodic Reward                 self.reward_per_episode
                Cumulative Reward               self.reward_total
        '''
        self.steps_per_episode += self.num_instances_per_env
        self.step_count += self.num_instances_per_env
        self.temp_done_counter += int(self.dones[self.roles[0]]) #sum(self.dones[role] for role in self.roles)
        self.done_count += int(self.dones[self.roles[0]]) #sum([self.dones[role] for role in self.roles])
        for role in self.roles:
            # in case there's asymetric environment
            self.reward_per_step[role] += self.rewards[role]
            self.reward_per_episode[role] += self.rewards[role]
            self.reward_total[role] += self.reward_per_episode[role]

        self.display()
        return list(self.observations.values()), list(self.rewards.values()), list(self.dones.values()), {}

    def get_metrics(self, episodic=True):
        '''MultiAgent Metrics'''
        metrics = {role:self.get_role_metrics(role, episodic) for ix, role in enumerate(self.roles)}
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

    def is_done(self):
        return self.steps_per_episode >= self.episode_max_length

    def _clean_actions(self, role, role_actions):
        '''
            Keep Discrete Actions as a One-Hot encode
        '''
        if self.env.discrete_action_space:
            # actions = np.array([ [np.argmax(_act)] for _act in role_actions ])
            actions = np.array(role_actions)
        elif type(role_actions) != np.ndarray:
            actions = np.array(role_actions)
        return actions

    def get_action_space_from_env(self, agent_action_space):
        '''All Action Spaces are Discrete - unless new environment is created by us'''
        if self.env.discrete_action_space:
            action_space = {
                'discrete': (agent_action_space.n,),
                'continuous': 0,
                'param': 0,
                'acs_space': (agent_action_space.n,)
            }
        else:
            assert "Continuous Action Space for the Particle Environment is not implemented"
            # all scenarios have discrete action space
            # action_space = {
            #     'discrete': agent_action_space.n,
            #     'continuous': 0,
            #     'param': 0,
            #     'acs_space': agent_action_space.n
            # }
        return action_space

    def get_observation_space_from_env(self, agent_obs_space):
        '''All Obs Spaces are Continuous - unless new environment is implemented'''
        observation_space = 1
        if agent_obs_space.shape != ():
            for i in range(len(agent_obs_space.shape)):
                observation_space *= agent_obs_space.shape[i]
        else:
            observation_space = agent_obs_space.n
        assert observation_space > 1, "Error processing Obs space? got {}".format(agent_obs_space)
        return observation_space

    def get_observations(self):
        return list(self.observations.values())

    def get_actions(self):
        return list(self.actions.values())

    def get_rewards(self):
        return list(self.rewards.values())

    def get_reward_episode(self, roles=True):
        return self.reward_per_episode

    def display(self):
        if self.render:
            self.env.render()

    def close(self):
        self.env.close()
        delattr(self, 'env')

    def debug(self):
        pass
