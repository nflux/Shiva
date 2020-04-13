import numpy as np
import copy
import torch
from collections.abc import Iterable

from torch.nn.functional import softmax
from shiva.utils import Noise as noise
from shiva.helpers.calc_helper import np_softmax
from shiva.agents.MADDPGAgent import MADDPGAgent
from shiva.algorithms.Algorithm import Algorithm
from shiva.helpers.networks_helper import perturb_optimizer
from shiva.networks.DynamicLinearNetwork import DynamicLinearNetwork
from shiva.helpers.misc import one_hot_from_logits

from itertools import permutations
from functools import partial

class MADDPGAlgorithm(Algorithm):
    def __init__(self, observation_space: int, action_space: dict, configs: dict):
        super(MADDPGAlgorithm, self).__init__(observation_space, action_space, configs)
        # self.actor_loss = [0 for _ in range(len(self.roles))]
        # self.critic_loss = [0 for _ in range(len(self.roles))]
        self.actor_loss = {}
        self.critic_loss = {}
        self.set_spaces(observation_space, action_space)
        '''
            Agent 1 and 2
            - Make sure actions/obs per agent are in the same indices in the buffer - don't sure how (I'm sure they come in the same order.. could be double checked)

            Methods
            Option 1 - critics
                Each agent has it's own critic, order should of data input to critic should be consistent
            Option 2 - discriminator
                Single critic with a one-hot encoding to correlate agents
                Expensive as it needs to find the correlation between the one-hot and all the obs/acs for each agent
                Increase size of network to let it learn more
            Option 3 - permutations
                Permute obs/acs and permute the reward being predicted
                Critic would think they are just one agent but looking at many datapoints (each agent is a diff datapoint)
                Agents should have the same Action Space
        '''
        self.critic_input_size = sum([self.action_space[role]['acs_space'] for role in self.roles]) + sum([self.observation_space[role] for role in self.roles])

        if self.method == "permutations":
            '''Single Local Critic'''
            self.critic = DynamicLinearNetwork(self.critic_input_size, 1, self.configs['Network']['critic']).to(self.device)
            self.target_critic = copy.deepcopy(self.critic)

            if hasattr(self.configs['Agent'], 'lr_range') and self.configs['Agent']['lr_range']:
                self.critic_learning_rate = np.random.uniform(self.configs['Agent']['lr_uniform'][0], self.configs['Agent']['lr_uniform'][1]) / np.random.choice( self.configs['Agent']['lr_factors'])
            else:
                self.critic_learning_rate = self.configs['Agent']['critic_learning_rate']

            self.optimizer_function = getattr(torch.optim, self.configs['Agent']['optimizer_function'])
            self.critic_optimizer = self.optimizer_function(params=self.critic.parameters(), lr=self.critic_learning_rate)
            self._update = self.update_permutes
        elif self.method == "critics":
            self._update = self.update_critics
        else:
            assert "MADDPG Method {} is not implemented".format(self.method)

        if not hasattr(self, 'update_iterations'):
            self.update_iterations = 1
        if self.configs['MetaLearner']['pbt']:
            self.resample_hyperparameters()

    def update(self, agents, buffer, step_count, episodic):
        self.log("", verbose_level=1)
        self.agents = agents
        for _ in range(self.update_iterations):
            self._update(agents, buffer, step_count, episodic)
        self.num_updates += self.update_iterations

    def update_permutes(self, agents: list, buffer: object, step_count: int, episodic=False):
        bf_states, bf_actions, bf_rewards, bf_next_states, bf_dones = buffer.sample(device=self.device)
        dones = bf_dones.bool()
        self.log("Obs {} Acs {} Rew {} NextObs {} Dones {}".format(bf_states, bf_actions, bf_rewards, bf_next_states, dones), verbose_level=4)
        self.log("FROM BUFFER Shapes Obs {} Acs {} Rew {} NextObs {} Dones {}".format(bf_states.shape, bf_actions.shape, bf_rewards.shape, bf_next_states.shape, bf_dones.shape), verbose_level=3)
        self.log("FROM BUFFER Types Obs {} Acs {} Rew {} NextObs {} Dones {}".format(bf_states.dtype, bf_actions.dtype, bf_rewards.dtype, bf_next_states.dtype, bf_dones.dtype), verbose_level=3)

        '''Transform buffer actions to a one hot or softmax if needed'''
        # for ix, (role, action_space) in enumerate(self.action_space.items()):
        #     if action_space['type'] == 'discrete':
        #         pass
        #         # no need if buffer stored one hot encodings
        #         # bf_actions[:, :, :] = one_hot_from_logits(bf_actions[:, ix, :]) for ix in range(self.num_agents)
        #     else:
        #         # Ezequiel: curious if here is doing a second softmax?
        #         bf_actions[:, ix, :] = softmax(bf_actions[:, ix, :])

        # helper function
        def _permutate(data, p, dim):
            # apply permutation p along all dimensions
            # g.e. p = (2,0,1) for 3 agents
            if not torch.is_tensor(p):
                p = torch.LongTensor(p)
            data = data.clone().detach()
            for d in range(data.shape[dim]):
                data[d] = data[d][p]
            return data

        # self.log("States from Buff {}".format(bf_rewards.reshape(1, -1)))
        '''Do all permutations of experiences to concat for the 1 single critic'''
        possible_permutations = set(permutations(np.arange(len(agents))))
        # self.log('Updating {} on permutations {}'.format([str(agent) for agent in agents], possible_permutations))
        for perms_ix, perms in enumerate(possible_permutations):
            agent_ix = perms[0]
            agent = agents[agent_ix]

            permutate_f = partial(_permutate, p=perms, dim=0)
            states = permutate_f(bf_states.to(self.device))
            actions = permutate_f(bf_actions.to(self.device))
            rewards = permutate_f(bf_rewards.to(self.device))
            next_states = permutate_f(bf_next_states.to(self.device))
            dones = permutate_f(dones.to(self.device))
            dones_mask = torch.tensor(dones[:, 0, :], dtype=torch.bool).view(-1, 1).to(self.device)
            # self.log("Permuted States {} is {}".format(perms, states.reshape(1, -1)))
            '''Assuming all agents have the same obs_dim!'''
            batch_size, num_agents, obs_dim = states.shape
            _, _, acs_dim = actions.shape

            # Zero the gradient
            self.critic_optimizer.zero_grad()

            # The actions that target actor would do in the next state & concat actions
            if self.action_space[agent.role]['type'] == 'discrete':
                '''Assuming Discrete Action Space ONLY here - if continuous need to one-hot only the discrete side'''
                # this iteration might not be following the same permutation order - at least is from a different _agent.target_actor
                next_state_actions_target = torch.cat([one_hot_from_logits(agents[perms[_ix]].target_actor(next_states[:, _ix, :])) for _ix, _agent in enumerate(agents)], dim=1)
                # self.log('OneHot next_state_actions_target {}'.format(next_state_actions_target))
            elif self.action_space[agent.role]['type'] == 'continuous':
                next_state_actions_target = torch.cat([agents[perms[_ix]].target_actor(next_states[:, _ix, :]) for _ix, _agent in enumerate(agents)], dim=1)

            Q_next_states_target = self.target_critic(torch.cat( [next_states.reshape(batch_size, num_agents*obs_dim).float(), next_state_actions_target.float()] , dim=1))
            # self.log('Q_next_states_target {}'.format(Q_next_states_target.shape))
            Q_next_states_target[dones_mask] = 0.0
            # self.log('Q_next_states_target {}'.format(Q_next_states_target.shape))
            # self.log('rewards {}'.format(rewards))
            # Use the Bellman equation.
            # Reward to predict is always index 0 from the already permuted rewards array
            y_i = rewards[:, 0, :] + self.gamma * Q_next_states_target
            # self.log("Rewards Agent ID {} {}".format(ix, rewards[:, 0, :].view(1, -1)))
            # self.log('y_i {}'.format(y_i.shape))

            # Get Q values of the batch from states and actions.
            Q_these_states_main = self.critic(torch.cat([states.reshape(batch_size, num_agents*obs_dim).float(), actions.reshape(batch_size, num_agents*acs_dim).float()], dim=1))
            # self.log('Q_these_states_main {}'.format(Q_these_states_main))

            # Calculate the loss.
            critic_loss = self.loss_calc(y_i.detach(), Q_these_states_main)
            # Backward propagation!
            critic_loss.backward()
            # Update the weights in the direction of the gradient.
            self.critic_optimizer.step()
            # Tensorboard
            self.critic_loss[agent.id] = critic_loss.item()

            '''
                Training the Actors
            '''

            # Zero the gradients
            for _agent in agents:
                _agent.actor_optimizer.zero_grad()
            # agent.actor_optimizer.zero_grad()

            # Get the actions the main actor would take from the initial states
            if self.action_space[agent.role]['type'] == "discrete":
                '''Option 1: grab new actions only for the current agent and keep original ones from the others..'''
                # current_state_actor_actions = actions
                # current_state_actor_actions[:, 0, :] = agent.actor(states[:, 0, :].float(), gumbel=True)
                '''Option 2: grab new actions from every agent: this might be destabilizer'''
                current_state_actor_actions = torch.cat([agents[perms[_ix]].actor(states[:, _ix, :].float(), gumbel=True) for _ix, _agent in enumerate(agents)], dim=1)
            elif self.action_space[agent.role]['type'] == "continuous":
                current_state_actor_actions = torch.cat([agents[perms[_ix]].actor(states[:, _ix, :].float(), gumbel=False) for _ix, _agent in enumerate(agents)], dim=1)

            # Calculate Q value for taking those actions in those states
            # self.log("current_state_actor_actions {}".format(current_state_actor_actions.shape))
            # self.log("states {}".format(states.shape))
            actor_loss_value = self.critic(torch.cat([states.reshape(batch_size, num_agents*obs_dim).float(), current_state_actor_actions.float()], dim=1))
            # entropy_reg = (-torch.log_softmax(current_state_actor_actions, dim=2).mean() * 1e-3)/1.0 # regularize using logs probabilities
            # penalty for going beyond the bounded interval
            param_reg = torch.clamp((current_state_actor_actions ** 2) - torch.ones_like(current_state_actor_actions), min=0.0).mean()
            # Make the Q-value negative and add a penalty if Q > 1 or Q < -1 and entropy for richer exploration
            actor_loss = -actor_loss_value.mean() + param_reg  # + entropy_reg
            # Backward Propogation!
            actor_loss.backward()
            # Update the weights in the direction of the gradient.
            agent.actor_optimizer.step()
            # Save actor loss for tensorboard
            self.actor_loss[agent.id] = actor_loss.item()

        '''
            After all Actor updates, soft update Target Networks
        '''
        for agent in agents:
            # Update Target Actor
            ac_state = agent.actor.state_dict()
            tgt_ac_state = agent.target_actor.state_dict()

            for k, v in ac_state.items():
                tgt_ac_state[k] = v * self.tau + (1 - self.tau) * tgt_ac_state[k]
            agent.target_actor.load_state_dict(tgt_ac_state)

        # Update Target Critic
        ct_state = self.critic.state_dict()
        tgt_ct_state = self.target_critic.state_dict()

        for k, v in ct_state.items():
            tgt_ct_state[k] = v * self.tau + (1 - self.tau) * tgt_ct_state[k]
        self.target_critic.load_state_dict(tgt_ct_state)

    def update_critics(self, agents: list, buffer: object, step_count: int, episodic=False):
        states, actions, rewards, next_states, dones = buffer.sample(device=self.device)
        '''Assuming same done flag for all agents on all timesteps'''
        dones_mask = torch.tensor(dones[:, 0, 0], dtype=torch.bool).view(-1, 1).to(self.device)
        # dones = dones.bool()
        # self.log("Obs {} Acs {} Rew {} NextObs {} Dones {}".format(states, actions, rewards, next_states, dones_mask))
        # self.log("FROM BUFFER Shapes Obs {} Acs {} Rew {} NextObs {} Dones {}".format(bf_states.shape, bf_actions.shape, bf_rewards.shape, bf_next_states.shape, bf_dones.shape), verbose_level=3)
        # self.log("FROM BUFFER Types Obs {} Acs {} Rew {} NextObs {} Dones {}".format(bf_states.dtype, bf_actions.dtype, bf_rewards.dtype, bf_next_states.dtype, bf_dones.dtype))

        '''Transform buffer actions to a one hot or softmax if needed'''
        # for ix, (role, action_space) in enumerate(self.action_space.items()):
        #     if action_space['type'] == 'discrete':
        #         pass
        #         # no need if buffer stored one hot encodings
        #         # bf_actions[:, :, :] = one_hot_from_logits(bf_actions[:, ix, :]) for ix in range(self.num_agents)
        #     else:
        #         # Ezequiel: curious if here is doing a second softmax?
        #         bf_actions[:, ix, :] = softmax(bf_actions[:, ix, :])

        # self.log("States from Buff {}".format(rewards.reshape(1, -1)))
        for agent_ix, agent in enumerate(self.agents):
            batch_size, num_agents, obs_dim = states.shape
            _, _, acs_dim = actions.shape

            # Zero the gradient
            agent.critic_optimizer.zero_grad()
            # The actions that target actor would do in the next state & concat actions
            if self.action_space[agent.role]['type'] == 'discrete':
                '''Assuming Discrete Action Space ONLY here - if continuous need to one-hot only the discrete side'''
                # this iteration might not be following the same permutation order - at least is from a different _agent.target_actor
                next_state_actions_target = torch.cat([one_hot_from_logits(_agent.target_actor(next_states[:, _ix, :])) for _ix, _agent in enumerate(agents)], dim=1)
                # self.log('OneHot next_state_actions_target {}'.format(next_state_actions_target))
            elif self.action_space[agent.role]['type'] == 'continuous':
                next_state_actions_target = torch.cat([_agent.target_actor(next_states[:, _ix, :]) for _ix, _agent in enumerate(agents)], dim=1)

            Q_next_states_target = agent.target_critic(torch.cat( [next_states.reshape(batch_size, num_agents*obs_dim).float(), next_state_actions_target.float()] , 1))
            # self.log('Q_next_states_target {}'.format(Q_next_states_target.shape))
            Q_next_states_target[dones_mask] = 0.0
            # self.log('Q_next_states_target {}'.format(Q_next_states_target.shape))
            # self.log('rewards {}'.format(rewards))
            # Use the Bellman equation.
            y_i = rewards[:, agent_ix, :] + self.gamma * Q_next_states_target
            # self.log("Rewards Agent ID {} {}".format(ix, rewards[:, 0, :].view(1, -1)))
            # self.log('y_i {}'.format(y_i.shape))

            # Get Q values of the batch from states and actions.
            Q_these_states_main = agent.critic(torch.cat([states.reshape(batch_size, num_agents*obs_dim).float(), actions.reshape(batch_size, num_agents*acs_dim).float()], 1))
            # self.log('Q_these_states_main {}'.format(Q_these_states_main))

            # Calculate the loss.
            agent_critic_loss = self.loss_calc(y_i.detach(), Q_these_states_main)
            # self.log('critic_loss {}'.format(self.critic_loss))
            # Backward propagation!
            agent_critic_loss.backward()
            # Update the weights in the direction of the gradient.
            agent.critic_optimizer.step()
            # Save for tensorboard
            self.critic_loss[agent.id] = agent_critic_loss.item()

            '''
                Training the Actors
            '''

            # Zero the gradients
            for _agent in agents:
                _agent.actor_optimizer.zero_grad()
            # Get the actions the main actor would take from the initial states
            if self.action_space[agent.role]['type'] == "discrete":
                current_state_actor_actions = torch.cat([_agent.actor(states[:, _ix, :].float(), gumbel=True) for _ix, _agent in enumerate(agents)], dim=1)
                # current_state_actor_actions = agent.actor(states[:, ix,s :].float(), gumbel=True)
            elif self.action_space[agent.role]['type'] == "continuous":
                current_state_actor_actions = torch.cat([_agent.actor(states[:, _ix, :].float()) for _ix, _agent in enumerate(agents)], dim=1)
                # current_state_actor_actions = agent.actor(states[:, ix, :].float())

            # Calculate Q value for taking those actions in those states
            actor_loss_value = agent.critic(torch.cat([states.reshape(batch_size, num_agents*obs_dim).float(), current_state_actor_actions.float()], dim=1))
            # actor_loss_value = self.critic(torch.cat([states[:, ix, :].float(), current_state_actor_actions[:, ix, :].float()], -1))
            # entropy_reg = (-torch.log_softmax(current_state_actor_actions, dim=2).mean() * 1e-3)/1.0 # regularize using logs probabilities
            # penalty for going beyond the bounded interval
            param_reg = torch.clamp((current_state_actor_actions ** 2) - torch.ones_like(current_state_actor_actions), min=0.0).mean()
            # Make the Q-value negative and add a penalty if Q > 1 or Q < -1 and entropy for richer exploration
            actor_loss = -actor_loss_value.mean() + param_reg  # + entropy_reg
            # Backward Propogation!
            actor_loss.backward()
            # Update the weights in the direction of the gradient.
            agent.actor_optimizer.step()
            # Save actor loss for tensorboard
            self.actor_loss[agent.id] = actor_loss.item()

        '''
            After all Actor updates, soft update Target Networks
        '''
        for agent in agents:
            # Update Target Actor
            ac_state = agent.actor.state_dict()
            tgt_ac_state = agent.target_actor.state_dict()

            for k, v in ac_state.items():
                tgt_ac_state[k] = v * self.tau + (1 - self.tau) * tgt_ac_state[k]
            agent.target_actor.load_state_dict(tgt_ac_state)

            # Update Target Critic
            ct_state = agent.critic.state_dict()
            tgt_ct_state = agent.target_critic.state_dict()

            for k, v in ct_state.items():
                tgt_ct_state[k] = v * self.tau + (1 - self.tau) * tgt_ct_state[k]
            agent.target_critic.load_state_dict(tgt_ct_state)

    def evolve(self, evol_config):
        pass

    def copy_weight_from_agent(self, evo_agent):
        # self.actor_learning_rate = evo_agent.actor_learning_rate
        self.critic_learning_rate = evo_agent.critic_learning_rate
        # self.epsilon = evo_agent.epsilon
        # self.noise_scale = evo_agent.noise_scale

        # self.actor.load_state_dict(evo_agent.actor.to(self.device).state_dict())
        # self.target_actor.load_state_dict(evo_agent.target_actor.to(self.device).state_dict())
        self.critic.load_state_dict(evo_agent.critic.to(self.device).state_dict())
        self.target_critic.load_state_dict(evo_agent.target_critic.to(self.device).state_dict())
        # self.actor_optimizer.load_state_dict(evo_agent.actor_optimizer.to(self.device).state_dict())
        self.critic_optimizer.load_state_dict(evo_agent.critic_optimizer.state_dict())

    def perturb_hyperparameters(self, perturb_factor):
        self.critic_learning_rate = self.critic_learning_rate * perturb_factor
        self.critic_optimizer = perturb_optimizer(self.critic_optimizer, {'lr': self.critic_learning_rate})

    def resample_hyperparameters(self):
        self.critic_learning_rate = np.random.uniform(self.configs['Agent']['lr_uniform'][0], self.configs['Agent']['lr_uniform'][1]) / np.random.choice(self.configs['Agent']['lr_factors'])
        self.critic_optimizer = perturb_optimizer(self.critic_optimizer, {'lr': self.critic_learning_rate})

    def create_agents(self):
        assert 'NotImplemented - this method could be creating all Roles agents at once'

    def create_agent_of_role(self, id, role):
        assert role in self.roles, "Invalid given role, got {} expected of {}".format(role, self.roles)
        self.configs['Agent']['role'] = role
        self.configs['Agent']['critic_input_size'] = self.critic_input_size
        self.agentCount += 1
        self.actor_loss[id] = 0
        self.critic_loss[id] = 0
        return MADDPGAgent(id, self.observation_space[role], self.action_space[role], self.configs['Agent'], self.configs['Network'])

    def set_spaces(self, observation_space, action_space):
        self.log("Got Obs Space {} and Acs Space {}".format(observation_space, action_space), verbose_level=3)
        if len(self.roles) == 1 and not isinstance(observation_space, Iterable):
            self.observation_space = {}
            self.observation_space[self.roles[0]] = observation_space
        self.set_action_space(action_space)

    def set_action_space(self, roles_action_space):
        self.action_space = {}
        for role in self.roles:
            if role not in roles_action_space:
                '''Here is DDPG collapse because we are running a Gym (or similar with single agent) - specifically if the Env doesn't output a dict of roles->dimensions when calling for acs/obs spaces'''
                roles_action_space[role] = roles_action_space
            if roles_action_space[role]['continuous'] == 0:
                roles_action_space[role]['type'] = 'discrete'
            elif roles_action_space[role]['discrete'] == 0:
                roles_action_space[role]['type'] = 'continuous'
            else:
                assert "Parametrized not supported yet"
            self.action_space[role] = roles_action_space[role]

    def save_central_critic(self, agent):
        # All Agents will host a copy of the central critic to enable evolution
        agent.critic.load_state_dict(self.critic.state_dict())
        agent.target_critic.load_state_dict(self.target_critic.state_dict())
        agent.critic_optimizer.load_state_dict(self.critic_optimizer.state_dict())
        agent.num_updates = self.get_num_updates()

    def load_central_critic(self, agent):
        self.critic.load_state_dict(agent.critic.state_dict())
        self.target_critic.load_state_dict(agent.target_critic.state_dict())
        self.critic_optimizer.load_state_dict(agent.critic_optimizer.state_dict())
        self.num_updates = agent.num_updates

    def get_metrics(self, episodic, agent_id):
        if not episodic:
            '''Step metrics'''
            metrics = []
        else:
            metrics = [
                ('Algorithm/Actor_Loss', self.actor_loss[agent_id]),
                ('Algorithm/Critic_Loss', self.critic_loss[agent_id])
            ]
            # if self.configs['MetaLearner']['pbt']:
            #     metrics += [('Agent/MADDPG/Critic_Learning_Rate', self.critic_learning_rate)]
        return metrics



    def __str__(self):
        return '<MADDPGAlgorithm(n_agents={}, num_updates={}, method={})>'.format(self.agentCount, self.num_updates, self.method)