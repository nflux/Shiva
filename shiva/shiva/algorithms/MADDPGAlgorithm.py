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

class MADDPGAlgorithm(DDPGAlgorithm):
    def __init__(self, observation_space: int, action_space: dict, configs: dict):
        super(MADDPGAlgorithm, self).__init__(observation_space, action_space, configs)
        self.actor_loss = torch.tensor(0)
        self.critic_loss = torch.tensor(0)
        self.set_action_space(action_space)

        critic_input = sum([self.action_space[role]['acs_space'] for role in self.roles]) + sum([self.observation_space[role] for role in self.roles])
        self.critic = DynamicLinearNetwork(critic_input, 1, self.configs['Network']['critic'])
        self.target_critic = copy.deepcopy(self.critic)
        self.optimizer_function = getattr(torch.optim, self.optimizer_function)
        self.critic_optimizer = self.optimizer_function(params=self.critic.parameters(), lr=self.critic_learning_rate)


    def update(self, agents: list, buffer: object, step_count: int, episodic=False):
        states, actions, rewards, next_states, dones = buffer.sample(device=self.device)
        dones = dones.bool()

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones_mask = torch.tensor(dones[:,0,0], dtype=torch.bool).view(-1, 1).to(self.device) # assuming done is the same for all agents

        '''Assuming all agents have the same obs_dim!'''
        batch_size, num_agents, obs_dim = states.shape
        _, _, acs_dim = actions.shape

        # print("Obs {} Acs {} Rew {} NextObs {} Dones {}".format(states, actions, rewards, next_states, dones_mask))
        print("Shapes Obs {} Acs {} Rew {} NextObs {} Dones {}".format(states.shape, actions.shape, rewards.shape, next_states.shape, dones_mask.shape))
        print("Types Obs {} Acs {} Rew {} NextObs {} Dones {}".format(states.dtype, actions.dtype, rewards.dtype, next_states.dtype, dones_mask.dtype))

        # Zero the gradient
        self.critic_optimizer.zero_grad()
        # The actions that target actor would do in the next state & concat actions
        '''Assuming Discrete Action Space ONLY here - if continuous need to one-hot only the discrete side'''
        next_state_actions_target = torch.cat([one_hot_from_logits(agent.target_actor(next_states[:, ix, :])) for ix, agent in enumerate(agents)], dim=-1)
        print('OneHot next_state_actions_target {}'.format(next_state_actions_target))

        Q_next_states_target = self.target_critic(torch.cat([next_states.reshape(batch_size, num_agents*obs_dim).float(), next_state_actions_target.float()], 1))
        print('Q_next_states_target {}'.format(Q_next_states_target))
        Q_next_states_target[dones_mask] = 0.0
        print('Q_next_states_target {}'.format(Q_next_states_target))

        # Use the Bellman equation.
        ''''''
        y_i = rewards + self.gamma * Q_next_states_target
        print('y_i {}'.format(y_i))

        for ix, (role, action_space) in enumerate(self.action_space.items()):
            if action_space['type'] == 'discrete':
                actions[:, ix, :] = one_hot_from_logits(actions[:, ix, :])
            else:
                # Ezequiel: curious if here is doing a second softmax?
                actions[:, ix, :] = softmax(actions[:, ix, :])

        # Get Q values of the batch from states and actions.
        Q_these_states_main = self.critic(torch.cat([states.reshape(batch_size, num_agents*obs_dim).float(), actions.reshape(batch_size, num_agents*acs_dim).float()], 1))
        print('Q_these_states_main {}'.format(Q_these_states_main))

        # Calculate the loss.
        critic_loss = self.loss_calc(y_i.detach(), Q_these_states_main)
        print('critic_loss {}'.format(critic_loss))
        # Backward propagation!
        critic_loss.backward()
        # Update the weights in the direction of the gradient.
        self.critic_optimizer.step()
        # Save critic loss for tensorboard
        self.critic_loss = critic_loss

        '''
            Training the Actors
        '''
        for ix, agent in enumerate(agents):
            # Zero the gradient
            agent.actor_optimizer.zero_grad()
            # Get the actions the main actor would take from the initial states
            if self.action_space[agent.role]['type'] == "discrete" or self.action_space[agent.role]['type'] == "parameterized":
                current_state_actor_actions = agent.actor(states[:, ix, :].float(), gumbel=True)
            else:
                current_state_actor_actions = agent.actor(states[:, ix, :].float())
            # Calculate Q value for taking those actions in those states

            '''
                Agent 1 and 2
                - Make sure actions/obs per agent are in the same indices
                
                Option 1:
                    Each agent has it's own critic, order should be consistent
                Option 2:
                    Single critic with a one-hot encoding to correlate (Discriminator)
                    Expensive as it needs to find the correlation between the one-hot and all the obs/acs
                    Increase size of network to let it learn more
                Option 3:
                    Permute obs/acs and permute the reward being predicted (Critic would think they are just one agent, but looking at many datapoints)
                
                Option 1: get historical actions
                Option 2: get current actions the agents would take (could be expensive)
                
                1 Actor loss
                
                Imagine agent with same reward function: 1 critic
                
                If one central critic and predict reward of each agent, need to permute the critic input as well on every agent iteration
                
                Discriminator: one hot encoding that changes on agent iteration, maybe good when the action space is different
                
                Basic approach:
                    Critic per agent who receives all obs and acs
            '''

            actor_loss_value = self.critic(torch.cat([states[:, ix, :].float(), current_state_actor_actions.float()], -1))
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
            Soft Target Network Updates
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

    def get_metrics(self, episodic=False):
        if not episodic:
            '''Step metrics'''
            metrics = []
        else:
            metrics = [
                ('Algorithm/Actor_Loss', self.actor_loss.item()),
                ('Algorithm/Critic_Loss', self.critic_loss.item())
            ]
        return metrics

    def __str__(self):
        return '<MADDPGAlgorithm>'