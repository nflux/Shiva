import numpy as np
import copy
import torch
from torch.nn.functional import softmax
from shiva.utils import Noise as noise
from shiva.helpers.calc_helper import np_softmax
from shiva.agents.MADDPGAgent import MADDPGAgent
from shiva.algorithms.DDPGAlgorithm import DDPGAlgorithm
from shiva.networks.DynamicLinearNetwork import DynamicLinearNetwork
from shiva.helpers.misc import one_hot_from_logits

from itertools import permutations
from functools import partial

class MADDPGAlgorithm(DDPGAlgorithm):
    def __init__(self, observation_space: int, action_space: dict, configs: dict):
        super(MADDPGAlgorithm, self).__init__(observation_space, action_space, configs)
        self.actor_loss = [torch.tensor(0) for _ in range(len(self.roles))]
        self.critic_loss = torch.tensor(0)
        self.set_action_space(action_space)

        critic_input = sum([self.action_space[role]['acs_space'] for role in self.roles]) + sum([self.observation_space[role] for role in self.roles])
        self.critic = DynamicLinearNetwork(critic_input, 1, self.configs['Network']['critic'])
        self.target_critic = copy.deepcopy(self.critic)
        self.optimizer_function = getattr(torch.optim, self.optimizer_function)
        self.critic_optimizer = self.optimizer_function(params=self.critic.parameters(), lr=self.critic_learning_rate)

        if self.method == "permutations":
            self.update = self.update_permutes
        else:
            assert "Only 'permutations' method is implemented for MADDPG"


    def update_permutes(self, agents: list, buffer: object, step_count: int, episodic=False):
        '''
            Agent 1 and 2
            - Make sure actions/obs per agent are in the same indices in the buffer - don't sure how

            Methods
            Option 1 - critics
                Each agent has it's own critic, order should be consistent
            Option 2 - discriminator
                Single critic with a one-hot encoding to correlate agents
                Expensive as it needs to find the correlation between the one-hot and all the obs/acs for each agent
                Increase size of network to let it learn more
            Option 3 - permutations
                Permute obs/acs and permute the reward being predicted
                Critic would think they are just one agent but looking at many datapoints (each agent is a diff datapoint)
                Agents should have the same Action Space
        '''

        '''Option 3'''
        bf_states, bf_actions, bf_rewards, bf_next_states, bf_dones = buffer.sample(device=self.device)
        dones = bf_dones.bool()
        # self.log("Obs {} Acs {} Rew {} NextObs {} Dones {}".format(states, actions, rewards, next_states, dones_mask))
        # self.log("FROM BUFFER Shapes Obs {} Acs {} Rew {} NextObs {} Dones {}".format(bf_states.shape, bf_actions.shape, bf_rewards.shape, bf_next_states.shape, bf_dones.shape))
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

        self.log("Rewards from Buff {}".format(bf_rewards))
        '''Do all permutations of experiences to concat for the 1 single critic'''
        possible_permutations = set(permutations(np.arange(len(agents))))
        self.log("will update with {} different permutations".format(len(possible_permutations)))
        for perms in possible_permutations:
            ix = perms[0]
            agent = agents[ix]
            self.log('Updating {} on permutation {}/{}'.format(agent, perms, set(permutations(np.arange(len(agents))))))
            permutate_f = partial(_permutate, p=perms, dim=0)
            states = permutate_f(bf_states.to(self.device))
            actions = permutate_f(bf_actions.to(self.device))
            rewards = permutate_f(bf_rewards.to(self.device))
            next_states = permutate_f(bf_next_states.to(self.device))
            dones_mask = torch.tensor(dones[:, 0, 0], dtype=torch.bool).view(-1, 1).to(self.device)
            self.log("Permuted {} is {}".format(perms, rewards))
            '''Assuming all agents have the same obs_dim!'''
            batch_size, num_agents, obs_dim = states.shape
            _, _, acs_dim = actions.shape

            # Zero the gradient
            self.critic_optimizer.zero_grad()

            # The actions that target actor would do in the next state & concat actions
            '''Assuming Discrete Action Space ONLY here - if continuous need to one-hot only the discrete side'''
            # this iteration might not be following the same permutation order - at least is from a different _agent.target_actor
            next_state_actions_target = torch.cat([one_hot_from_logits(_agent.target_actor(next_states[:, perms[_ix], :])) for _ix, _agent in enumerate(agents)], dim=1)
            # self.log('OneHot next_state_actions_target {}'.format(next_state_actions_target))

            Q_next_states_target = self.target_critic(torch.cat( [next_states.reshape(batch_size, num_agents*obs_dim).float(), next_state_actions_target.float()] , 1))
            # self.log('Q_next_states_target {}'.format(Q_next_states_target.shape))
            Q_next_states_target[dones_mask] = 0.0
            # self.log('Q_next_states_target {}'.format(Q_next_states_target.shape))
            # self.log('rewards {}'.format(rewards))
            # Use the Bellman equation.
            # Reward to predict is always index 0
            y_i = rewards[:, 0, :] + self.gamma * Q_next_states_target
            # self.log("Rewards Agent ID {} {}".format(ix, rewards[:, 0, :].view(1, -1)))
            # self.log('y_i {}'.format(y_i.shape))

            # Get Q values of the batch from states and actions.
            Q_these_states_main = self.critic(torch.cat([states.reshape(batch_size, num_agents*obs_dim).float(), actions.reshape(batch_size, num_agents*acs_dim).float()], 1))
            # self.log('Q_these_states_main {}'.format(Q_these_states_main))

            # Calculate the loss.
            self.critic_loss = self.loss_calc(y_i.detach(), Q_these_states_main)
            # self.log('critic_loss {}'.format(self.critic_loss))
            # Backward propagation!
            self.critic_loss.backward()
            # Update the weights in the direction of the gradient.
            self.critic_optimizer.step()

            '''
                Training the Actors
            '''

            # Zero the gradients
            { _agent.actor_optimizer.zero_grad() for _agent in agents }
            # Get the actions the main actor would take from the initial states
            if self.action_space[agent.role]['type'] == "discrete" or self.action_space[agent.role]['type'] == "parameterized":
                current_state_actor_actions = torch.cat([_agent.actor(states[:, perms[_ix], :].float(), gumbel=True) for _ix, _agent in enumerate(agents)], dim=1)
                # current_state_actor_actions = agent.actor(states[:, ix,s :].float(), gumbel=True)
            else:
                current_state_actor_actions = torch.cat([_agent.actor(states[:, perms[_ix], :].float()) for _ix, _agent in enumerate(agents)], dim=1)
                # current_state_actor_actions = agent.actor(states[:, ix, :].float())
            # Calculate Q value for taking those actions in those states
            actor_loss_value = self.critic(torch.cat([states.reshape(batch_size, num_agents*obs_dim).float(), current_state_actor_actions.float()], dim=1))
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
            self.actor_loss[ix] = actor_loss

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

    def create_agent_of_role(self, role):
        assert role in self.roles, "Invalid given role, got {} expected of {}".format(role, self.roles)
        self.configs['Agent']['role'] = role
        return MADDPGAgent(self.id_generator(), self.observation_space[role], self.action_space[role], self.configs['Agent'], self.configs['Network'])

    def set_action_space(self, role_action_space):
        self.action_space = {}
        for role in list(role_action_space.keys()):
            if role_action_space[role]['continuous'] == 0:
                role_action_space[role]['type'] = 'discrete'
            elif role_action_space[role]['discrete'] == 0:
                role_action_space[role]['type'] = 'continuous'
            else:
                assert "Parametrized not supported yet"
            self.action_space[role] = role_action_space[role]

    def get_metrics(self, episodic, agent_id):
        if not episodic:
            '''Step metrics'''
            metrics = []
        else:
            metrics = [
                ('Algorithm/Actor_Loss'.format(agent_id), self.actor_loss[agent_id].item()),
                ('Algorithm/Critic_Loss', self.critic_loss.item())
            ]
        return metrics

    def __str__(self):
        return '<MADDPGAlgorithm>'