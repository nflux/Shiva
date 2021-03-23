import numpy as np
import copy
import torch
from collections.abc import Iterable

from shiva.agents.MADDPGAgent import MADDPGAgent
from shiva.algorithms.Algorithm import Algorithm
from shiva.helpers.networks_helper import mod_optimizer
from shiva.networks.DynamicLinearNetwork import DynamicLinearNetwork
from shiva.helpers.misc import one_hot_from_logits
from shiva.helpers.torch_helper import normalize_branches
from typing import Dict, Tuple, List, Union, Any
from itertools import permutations
from functools import partial

torch.autograd.set_detect_anomaly(True)

class MADDPGAlgorithm(Algorithm):
    def __init__(self, observation_space: Dict[str, int], action_space: Dict[str, Dict[str, Tuple[Union[int]]]], configs: Dict[str, Any]) -> None:
        """
        This class follows the algorithmic update explained on Ryan Lowe et all paper: https://arxiv.org/abs/1706.02275
        We have implemented 2 different methods: one using permutations with a single critic (link to method) and other using multiple critics (link to method).
        See MADDPG config explanation for correct usage.

        Args:
            observation_space (Dict[str, int]):
            action_space (Dict[str, Dict[str, Tuple[int, ...]]]): This is the action space dictionary that our Environment wrappers output (link to Environment)
            configs (Dict[str, ...]): The global config used for the run
        """
        super(MADDPGAlgorithm, self).__init__(observation_space, action_space, configs)
        torch.manual_seed(self.manual_seed)
        np.random.seed(self.manual_seed)
        self.log(f"MANUAL SEED {self.manual_seed}")

        self.actor_loss = {}
        self.critic_loss = {}
        self._metrics = {}
        self.set_spaces(observation_space, action_space)
        '''
            Agent 1 and 2
            - Make sure actions/obs per agent are in the same indices in the buffer - don't sure how (I'm sure they come in the same order.. could be double checked)

            Methods
            Option 1 - critics
                Each agent has it's own critic, order of data input to critic should be consistent
            Option 2 - discriminator
                Single critic with a one-hot encoding to correlate agents
                Expensive as it needs to find the correlation between the one-hot and all the obs/acs for each agent
                Increase size of network to let it learn more
            Option 3 - permutations
                Permute obs/acs and permute the reward being predicted
                Critic would think they are just one agent but looking at many datapoints (each agent is a diff datapoint)
                Agents should have the same Action Space
        '''
        self.critic_input_size = 0
        for role in self.roles:
            self.critic_input_size += sum(self.action_space[role]['acs_space']) + self.observation_space[role]

        if self.method == "permutations":
            '''Single Local Critic'''
            self.critic = DynamicLinearNetwork(self.critic_input_size, 1, self.configs['Network']['critic']).to(self.device)
            self.target_critic = copy.deepcopy(self.critic)

            if hasattr(self.configs['Agent'], 'hp_random') and self.configs['Agent']['hp_random']:
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

        self.gumbel = partial(torch.nn.functional.gumbel_softmax, tau=1, hard=True, dim=-1)

    def update(self, agents, buffer, step_count, episodic) -> None:
        """
        Runs the corresponding update method for a number of times specified by the config.
        See MADDPG config explanation for the available update methods.

        Args:
            agents (List[MADDPGAgent]): List of agents to be updated
            buffer (MultiAgentTensorBuffer): only buffers that have a `sample()` method that returns a 5 elements tuple in the order specified by the MultiAgentTensorBuffer (link)
        Returns:
            None
        """
        self.agents = agents
        self._metrics = {agent.id:[] for agent in self.agents}
        for _ in range(self.update_iterations):
            self._update(agents, buffer, step_count, episodic)
        # self.num_updates += self.update_iterations

    def update_permutes(self, agents: list, buffer: object, step_count: int, episodic=False):
        """
        Update method when Algorithm section in config has attribute `method = "permutations"`
        This method performs permutations on the observations, actions and target reward for a single central augmented critic.
        By doing permutations we can reuse a same transition datapoint as many times as permutations are there possible with the number of actors being updates.

        Note that this method only works when the agents have the same observation space, action spaces and reward function.

        Args:
            agents(List[MADDPGAgent]):
            buffer(MultiAgentTensorBuffer): only buffers that have a `sample()` method that returns a 5 elements tuple in the order specified by the MultiAgentTensorBuffer (link)

        Returns:
            None

        """
        bf_states, bf_actions, bf_rewards, bf_next_states, bf_dones, bf_actions_mask, bf_next_actions_mask = buffer.sample(device=self.device)
        dones = bf_dones.bool()

        # self.log(f"Obs {bf_states}", verbose_level=2)
        # self.log(f"Acs {bf_actions}", verbose_level=2)
        # self.log(f"Rew {bf_rewards}", verbose_level=2)
        # self.log(f"NextObs {bf_next_states}", verbose_level=2)
        # self.log(f"Done {bf_dones}", verbose_level=2)

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

        '''Do all permutations of experiences to concat for the 1 single critic'''
        possible_permutations = set(permutations(np.arange(len(agents))))
        for perms_ix, perms in enumerate(possible_permutations):
            agent_ix = perms[0]
            agent = agents[agent_ix]

            permutate_f = partial(_permutate, p=perms, dim=0)
            states = permutate_f(bf_states.to(self.device).float())
            actions = permutate_f(bf_actions.to(self.device))
            rewards = permutate_f(bf_rewards.to(self.device))
            next_states = permutate_f(bf_next_states.to(self.device).float())
            dones = permutate_f(dones.to(self.device))
            actions_mask = permutate_f(bf_actions_mask.to(self.device))
            next_actions_mask = permutate_f(bf_next_actions_mask.to(self.device))

            dones_mask = dones[:, 0, :].view(-1, 1).to(self.device)
            # print("States", states)
            # print("Actions", actions)
            # print("AcsMask", actions_mask)

            '''Assuming all agents have the same obs_dim!'''
            batch_size, num_agents, obs_dim = states.shape
            _, _, acs_dim = actions.shape

            '''
                Train the Critic
            '''
            # Zero the gradient
            self.critic_optimizer.zero_grad()

            # # The actions that target actor would do in the next state & concat actions
            # if self.action_space[agent.role]['type'] == 'discrete':
            #     '''Assuming Discrete Action Space ONLY here - if continuous need to one-hot only the discrete side'''
            #     next_state_actions_target = torch.cat([one_hot_from_logits(agents[perms[_ix]].target_actor(next_states[:, _ix, :])) for _ix, _agent in enumerate(agents)], dim=1)
            # elif self.action_space[agent.role]['type'] == 'continuous':
            #     next_state_actions_target = torch.cat([agents[perms[_ix]].target_actor(next_states[:, _ix, :]) for _ix, _agent in enumerate(agents)],dim=1)
            aux = []
            for _ix in range(len(agents)):
                a = agents[perms[_ix]]
                logits = a.target_actor(next_states[:, _ix, :])
                # Apply mask
                # logits = torch.where(next_actions_mask[:, _ix, :], torch.tensor(0.).to(self.device), logits)
                logits[next_actions_mask[:, _ix, :]] = 0
                # Renormalize each branch
                # logits = normalize_branches(logits, a.actor_output, one_hot_from_logits if a.action_space == "discrete" else None)
                _cum_ix = 0
                for ac_dim in a.actor_output:
                    _branch_action = logits[:, _cum_ix:ac_dim+_cum_ix].clone().abs()
                    _normalized_actions = _branch_action / _branch_action.sum(-1).reshape(-1, 1)
                    if a.action_space == 'continuous':
                        logits[:, _cum_ix:ac_dim+_cum_ix] = _normalized_actions
                    else:
                        logits[:, _cum_ix:ac_dim+_cum_ix] = one_hot_from_logits(_normalized_actions)
                    _cum_ix += ac_dim
                aux += [logits]
            next_state_actions_target = torch.cat(aux, dim=1)

            Q_next_states_target = self.target_critic(torch.cat( [next_states.reshape(batch_size, num_agents*obs_dim).float(), next_state_actions_target.float()] , dim=1))
            Q_next_states_target[dones_mask] = 0.0
            # Use the Bellman equation.
            # Reward to predict is always index 0 from the already permuted rewards array
            y_i = rewards[:, 0, :] + self.gamma * Q_next_states_target

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
            # self.critic_loss[agent.id] = critic_loss.item()

            '''
                Training the Actors
            '''

            # Zero the gradients
            for _agent in agents:
                _agent.actor_optimizer.zero_grad()
            # agent.actor_optimizer.zero_grad()

            # Get the actions the main actor would take from the initial states
            # if self.action_space[agent.role]['type'] == "discrete":
            #     '''Option 1: grab new actions only for the current agent and keep original ones from the others..'''
            #     # current_state_actor_actions = actions
            #     # current_state_actor_actions[:, 0, :] = agent.actor(states[:, 0, :].float(), gumbel=True)
            #     '''Option 2: grab new actions from every agent: this might be destabilizer'''
            #     current_state_actor_actions = torch.cat([agents[perms[_ix]].actor(states[:, _ix, :].float(), gumbel=True) for _ix, _agent in enumerate(agents)], dim=1)
            # elif self.action_space[agent.role]['type'] == "continuous":
            #     current_state_actor_actions = torch.cat([agents[perms[_ix]].actor(states[:, _ix, :].float()) for _ix, _agent in enumerate(agents)], dim=1)
            aux = []
            for _ix in range(len(agents)):
                a = agents[perms[_ix]]
                logits = a.actor(states[:, _ix, :])
                # Apply mask
                # logits = torch.where(actions_mask[:, _ix, :], torch.tensor([0.]).to(self.device), logits)
                logits[actions_mask[:, _ix, :]] = 0
                # Renormalize each branch
                # logits = normalize_branches(logits, a.actor_output, self.gumbel if a.action_space == "discrete" else None)
                _cum_ix = 0
                for ac_dim in a.actor_output:
                    _branch_action = logits[:, _cum_ix:ac_dim+_cum_ix].clone().abs()
                    if a.action_space == 'continuous':
                        _normalized_actions = _branch_action / _branch_action.sum(-1).reshape(-1, 1)
                        logits[:, _cum_ix:ac_dim+_cum_ix] = _normalized_actions
                    else:
                        logits[:, _cum_ix:ac_dim+_cum_ix] = self.gumbel(_branch_action)
                    _cum_ix += ac_dim
                aux += [logits]
            current_state_actor_actions = torch.cat(aux, dim=1)

            # Calculate Q value for taking those actions in those states
            # self.log(f"current_state_actor_actions {current_state_actor_actions.shape} {current_state_actor_actions}")
            # self.log("states {}".format(states.shape))
            actor_loss_value = self.critic(torch.cat([states.reshape(batch_size, num_agents*obs_dim).float(), current_state_actor_actions.float()], dim=1))
            # entropy_reg = (-torch.log_softmax(current_state_actor_actions, dim=2).mean() * 1e-3)/1.0 # regularize using logs probabilities
            # penalty for going beyond the bounded interval
            # param_reg = torch.clamp((current_state_actor_actions ** 2) - torch.ones_like(current_state_actor_actions), min=0.0).mean()
            # Make the Q-value negative and add a penalty if Q > 1 or Q < -1 and entropy for richer exploration
            actor_loss = -actor_loss_value.mean()# + param_reg  # + entropy_reg
            # Backward Propogation!
            actor_loss.backward()
            # Update the weights in the direction of the gradient.
            agent.actor_optimizer.step()
            # Save actor loss for tensorboard
            # self.actor_loss[agent.id] = actor_loss.item()

            self.num_updates += 1
            self._metrics[agent.id] += [('Algorithm/Actor_Loss', actor_loss.item(), self.num_updates)]
            self._metrics[agent.id] += [('Algorithm/Critic_Loss', critic_loss.item(), self.num_updates)]
            self._metrics[agent.id] += [('Agent/Central_Critic_Learning_Rate', self.critic_learning_rate, self.num_updates)]

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
        """
        Method to be revised and tested!
        Update method when Algorithm section in config has attribute `method = "critics"`
        With this method each actor will have it's own augmented critic. It must be used in the case where the actors have different observation/action spaces and reward function.
        Each actor will have it's own critic.

        Args:
            agents (List[MADDPGAgent]): List of agents to be updated
            buffer (MultiAgentTensorBuffer): only buffers that have a `sample()` method that returns a 5 elements tuple in the order specified by the MultiAgentTensorBuffer (link)

        Returns:
            None
        """
        states, actions, rewards, next_states, dones = buffer.sample(device=self.device)
        '''Assuming same done flag for all agents on all timesteps'''
        dones_mask = torch.tensor(dones[:, 0, 0], dtype=torch.bool).view(-1, 1).to(self.device)
        # dones = dones.bool()
        # self.log("Obs {} Acs {} Rew {} NextObs {} Dones {}".format(states, actions, rewards, next_states, dones_mask))
        # self.log("FROM BUFFER Shapes Obs {} Acs {} Rew {} NextObs {} Dones {}".format(bf_states.shape, bf_actions.shape, bf_rewards.shape, bf_next_states.shape, bf_dones.shape), verbose_level=3)
        # self.log("FROM BUFFER Types Obs {} Acs {} Rew {} NextObs {} Dones {}".format(bf_states.dtype, bf_actions.dtype, bf_rewards.dtype, bf_next_states.dtype, bf_dones.dtype))

        # self.log("States from Buff {}".format(rewards.reshape(1, -1)))

        self.num_updates += 1
        for agent_ix, agent in enumerate(self.agents):
            batch_size, num_agents, obs_dim = states.shape
            _, _, acs_dim = actions.shape

            '''
                Train the Critic
            '''
            # Zero the gradient
            agent.critic_optimizer.zero_grad()
            # The actions that target actor would do in the next state & concat actions
            if self.action_space[agent.role]['type'] == 'discrete':
                '''Assuming Discrete Action Space ONLY here - if continuous need to one-hot only the discrete side'''
                # this iteration might not be following the same permutation order - at least is from a different _agent.target_actor
                # next_state_actions_target = torch.cat([one_hot_from_logits(_agent.target_actor(next_states[:, _ix, :])) for _ix, _agent in enumerate(agents)], dim=1)
                next_state_actions_target = torch.cat([one_hot_from_logits(agents[perms[_ix]].target_actor(next_states[:, _ix, :])) for _ix, _agent in enumerate(agents)], dim=1)
                # self.log('OneHot next_state_actions_target {}'.format(next_state_actions_target))
            elif self.action_space[agent.role]['type'] == 'continuous':
                # next_state_actions_target = torch.cat([_agent.target_actor(next_states[:, _ix, :]) for _ix, _agent in enumerate(agents)], dim=1)
                next_state_actions_target = torch.cat([agents[perms[_ix]].target_actor(next_states[:, _ix, :]) for _ix, _agent in enumerate(agents)], dim=1)

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
                # current_state_actor_actions = torch.cat([_agent.actor(states[:, _ix, :].float(), gumbel=True) for _ix, _agent in enumerate(agents)], dim=1)
                current_state_actor_actions = torch.cat([agents[perms[_ix]].actor(states[:, _ix, :].float(), gumbel=True) for _ix, _agent in enumerate(agents)], dim=1)
            elif self.action_space[agent.role]['type'] == "continuous":
                current_state_actor_actions = torch.cat([_agent.actor(states[:, _ix, :].float()) for _ix, _agent in enumerate(agents)], dim=1)
                current_state_actor_actions = torch.cat([agents[perms[_ix]].actor(states[:, _ix, :].float()) for _ix, _agent in enumerate(agents)], dim=1)

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

            self._metrics[agent.id] += [('Algorithm/Actor_Loss', actor_loss.item(), self.num_updates)]
            self._metrics[agent.id] += [('Algorithm/Critic_Loss', critic_loss.item(), self.num_updates)]
            self._metrics[agent.id] += [('Agent/Central_Critic_Learning_Rate', self.critic_learning_rate, self.num_updates)]
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

    def copy_hyperparameters(self, evo_agent):
        """
        Copies the critic hyperparameters from the `evo_agent` to the local
        Only the critic learning rate is being copied for now.

        Args:
            evo_agent: (MADDPGAgent):

        Returns:
            None

        """
        self.critic_learning_rate = evo_agent.critic_learning_rate
        self._update_optimizer()

    def copy_weight_from_agent(self, evo_agent):
        """
        Copies the weights of the critic network, target critic network and critic optimizer from the `evo_agent` to the local

        Args:
            evo_agent (MADDPGAgent):

        Returns:
            None

        """
        self.critic.load_state_dict(evo_agent.critic.to(self.device).state_dict())
        self.target_critic.load_state_dict(evo_agent.target_critic.to(self.device).state_dict())
        self.critic_optimizer.load_state_dict(evo_agent.critic_optimizer.state_dict())

    def perturb_hyperparameters(self, perturb_factor):
        """
        Perturbs the local critic learning rate by the `perturb_factor`

        Args:
            perturb_factor (float): float value for which the local `critic_learning_rate` is multiplied

        Returns:
            None
        """
        self.critic_learning_rate *= perturb_factor
        self._update_optimizer()

    def resample_hyperparameters(self):
        """
        Resamples a new `critic_learning_rate` using a [don't know how to explain this function for which we sample a new LR]

        Returns:
            None

        """
        self.critic_learning_rate = np.random.uniform(self.configs['Agent']['lr_uniform'][0], self.configs['Agent']['lr_uniform'][1]) / np.random.choice(self.configs['Agent']['lr_factors'])
        self._update_optimizer()

    def decay_learning_rate(self):
        """
        Decays the local `critic_learning_rate` using `configs['Agent']['lr_decay']['factor']`

        Returns:
            None

        """
        self.critic_learning_rate *= self.configs['Agent']['lr_decay']['factor']
        self.log(f"Decay Critic LR {self.critic_learning_rate}", verbose_level=3)
        self._update_optimizer()

    def restore_learning_rate(self):
        """
        Opposite to `decay_learning_rate()`, restores the `critic_learning_rate`_ using `configs['Agent']['lr_decay']['factor']` until the maximum learning rate is reached

        Returns:
            None

        """
        if self.critic_learning_rate < self.configs['Agent']['critic_learning_rate']:
            self.critic_learning_rate /= self.configs['Agent']['lr_decay']['factor']
            self.log(f"Increment Critic LR {self.critic_learning_rate}", verbose_level=3)
        else:
            self.critic_learning_rate = self.configs['Agent']['critic_learning_rate']
            self.log(f"Critic LR Restored {self.critic_learning_rate}", verbose_level=3)
        self._update_optimizer()

    def _update_optimizer(self):
        self.critic_optimizer = mod_optimizer(self.critic_optimizer, {'lr': self.critic_learning_rate})

    def save_central_critic(self, agent):
        """
        This function is used to save the central critic (hosted in the algorithm) in the agents. This enables the critic evolution across learners.

        Args:
            agent (MADDPGAgent): agent that is gonna host a copy of our local critic network

        Returns:
            None

        """
        # All Agents will host a copy of the central critic to enable evolution
        agent.critic_learning_rate = self.critic_learning_rate
        agent.critic.load_state_dict(self.critic.state_dict())
        agent.target_critic.load_state_dict(self.target_critic.state_dict())
        agent.critic_optimizer.load_state_dict(self.critic_optimizer.state_dict())
        agent.num_updates = self.get_num_updates()

    def load_central_critic(self, agent):
        """
        This function loads the critic from the `agent` input to the local critic hosted in this algorithm.

        Args:
            agent (MADDPGAgent): agent from which we are gonna make a copy of the critic network

        Returns:
            None

        """
        self.critic_learning_rate = agent.critic_learning_rate
        self.critic.load_state_dict(agent.critic.state_dict())
        self.target_critic.load_state_dict(agent.target_critic.state_dict())
        self.critic_optimizer.load_state_dict(agent.critic_optimizer.state_dict())
        self.num_updates = agent.num_updates

    def create_agents(self):
        """
        Not implemented

        Returns:

        """
        assert 'NotImplemented - this method could be creating all Roles agents at once'

    def create_agent_of_role(self, id, role):
        """
        Creates a new MADDPG

        Args:
            id (int): unique ID for the agent
            role (str): role name of the agent

        Returns:
            MADDPGAgent

        """
        assert role in self.roles, "Invalid given role, got {} expected of {}".format(role, self.roles)
        self.configs['Agent']['role'] = role
        self.configs['Agent']['critic_input_size'] = self.critic_input_size
        new_agent = MADDPGAgent(id, self.observation_space[role], self.action_space[role], self.configs)
        self.add_agent(new_agent)
        return new_agent

    def add_agent(self, agent):
        """
        Function currently being used when loading agents. This way the algorithm can have a pointer to them.
        May be deprecated.

        Args:
            agent (MADDPGAgent):

        Returns:
            None

        """
        self.agentCount += 1
        self.actor_loss[agent.id] = 0
        self.critic_loss[agent.id] = 0

    def add_agents(self, agents:list):
        """
        Function currently being used when loading agents. This way the algorithm can have a pointer to them.
        May be deprecated.

        Args:
            agents (List[MADDPGAgent]):

        Returns:
            None

        """
        assert len(agents) > 0, "Empty list of agents to load"
        for a in agents:
            self.add_agent(a)
        self.load_central_critic(agents[0]) # for a central critic, all agents host a copy of the algs critic so we can grab any of them

    def set_spaces(self, observation_space, action_space):
        """
        Function called during initialization to pre-process the observation and action space. Calls `set_action_space()`

        Args:
            observation_space (Dict[str, int]):
            action_space (Dict[str, Dict[str, tuple]]): This is the action space dictionary that our Environment wrappers output (link to Environment)

        Returns:
            None

        """
        self.log("Got Obs Space {} and Acs Space {}".format(observation_space, action_space), verbose_level=3)
        if len(self.roles) == 1 and not isinstance(observation_space, Iterable):
            self.observation_space = {}
            self.observation_space[self.roles[0]] = observation_space
        self.set_action_space(action_space)

    def set_action_space(self, roles_action_space):
        """
        Function called during initialization to pre-process the action space.

        Args:
            roles_action_space (Dict[str, Dict[str, tuple]]):

        Returns:
            None

        """
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

    def get_metrics(self, episodic, agent_id):
        """
        Get current metrics from the MADDPG algorithm, specifically actor and critic loss.

        Args:
            agent_id (int): Agent ID for which we want to the collected metrics

        Returns:
            Returns a list of tuples for the Tensorboard.
            Each tuple is of the form (metric_name, y_value, x_value)

        Examples:
            >>> algorithm.get_metrics(agent_id=1)
            [('Critic_loss', 0.002, 10), ('Actor_loss', 0.11, 10), ('Critic_loss', 0.05, 11), ('Actor_loss', 0.6, 11)]

        """
        return self._metrics[agent_id] if agent_id in self._metrics else []
        # if not episodic:
        #     '''Step metrics'''
        #     metrics = []
        # else:
        #     metrics = [
        #         ('Algorithm/Actor_Loss', self.actor_loss[agent_id], self.num_updates),
        #         ('Algorithm/Critic_Loss', self.critic_loss[agent_id], self.num_updates),
        #         ('Agent/Central_Critic_Learning_Rate', self.critic_learning_rate, self.num_updates)
        #     ]
        # return metrics

    def __str__(self):
        return '<MADDPGAlgorithm(n_agents={}, num_updates={}, method={}, device={})>'.format(self.agentCount, self.num_updates, self.method, self.device)
