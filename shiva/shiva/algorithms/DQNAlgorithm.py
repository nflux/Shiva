import numpy as np
import torch
import random

from shiva.algorithms.Algorithm import Algorithm
from shiva.agents.DQNAgent import DQNAgent


class DQNAlgorithm(Algorithm):
    def __init__(self, obs_space, acs_space, configs):
        """ DQN Algorithm

        This class is based on the Deep Q-Network paper.

            Args:
                obs_space:
                acs_space:
        """
        super(DQNAlgorithm, self).__init__(obs_space, acs_space, configs)
        torch.manual_seed(self.manual_seed)
        np.random.seed(self.manual_seed)
        self.acs_space = acs_space[self.roles[0]]
        self.obs_space = obs_space
        self.loss = torch.tensor([0])
        self._metrics = {}

    def update(self, agents, buffer, step_count, episodic):
        # self.log("Start update # {}".format(self.num_updates))
        self.agents = agents
        self._metrics = {agent.id: [] for agent in self.agents}
        for _ in range(self.update_iterations):
            self._update(agents, buffer, step_count, episodic)
        self.num_updates += self.update_iterations

    def _update(self, agent, buffer, step_n, episodic=False):
        """
            Implementation
                1) Calculate what the current expected Q val from each sample on the replay buffer would be
                2) Calculate loss between current and past reward
                3) Optimize
                4) If agent steps reached C, update agent.target network

            Args:
                agent:       Agent reference who we are updating
                buffer:      Buffer reference to sample from
                step_n:      Current step number or done_count when doing episodic updates!!!!
                episodic:    Flag indicating if update is episodic
        """
        self.agent = agent[0] if type(agent) == list else agent

        # if episodic: # for DQN to do step-wise updates only
        # if not episodic: # for DQN to do episodic update
        #     return

        try:
            '''For MultiAgentTensorBuffer - 1 Agent only here'''
            states, actions, rewards, next_states, dones = buffer.sample(agent_id=self.agent.id, device=self.device)
            dones = dones.bool()
        except:
            states, actions, rewards, next_states, dones = buffer.sample(device=self.device)

        # print('from buffer Obs {} Acs {} Rew {} NextObs {} Dones {}:'.format(states[:3], actions[:3], rewards[:3], next_states[:3], dones[:3]))
        # # print('from buffer Acs: {} \n'.format(actions))

        self.agent.critic_optimizer.zero_grad()
        # 1) GRAB Q_VALUE(s_j, a_j)
        input_v = torch.cat([states, actions], dim=-1)
        state_action_values = self.agent.critic(input_v)

        # 2) GRAB MAX[Q_HAT_VALUES(s_j+1)]
        # For the observations s_j+1, select an action using the Policy and calculate Q values of those using the Target net
        target_next_state_actions = torch.tensor([self.agent.get_action_target(s_i) for s_i in next_states]).to(self.device)
        input_v = torch.cat([next_states, target_next_state_actions], dim=-1)
        next_state_values = self.agent.target_critic(input_v)

        # 3) Overwrite 0 on all next_state_values where they were termination states
        next_state_values[dones] = 0.0

        # 4) Detach magic
        # We detach the value from its computation graph to prevent gradients from flowing into the neural network used to calculate Q approximation next states.
        # Without this our backpropagation of the loss will start to affect both predictions for the current state and the next state.
        next_state_values = next_state_values.detach()
        expected_state_action_values = next_state_values * self.gamma + rewards

        # print(done_mask.shape, next_state_values.shape)
        # print(expected_state_action_values.shape, rewards_v.shape)
        self.loss = self.loss_calc(state_action_values, expected_state_action_values)

        # The only issue is referencing the learner from here for the first parameter
        # shiva.add_summary_writer(, agent, 'Loss per Step', loss_v, step_n)

        self.loss.backward()
        self.agent.critic_optimizer.step()

        if step_n % self.c == 0:
            # print('update target')
            self.agent.target_critic.load_state_dict(self.agent.critic.state_dict()) # Assuming is PyTorch!
        self.num_updates += 1
        self._metrics[self.agent.id] += [('Algorithm/Critic_Loss', self.loss.item(), self.num_updates)]

# =======
#             # try:
#             #     '''For MultiAgentTensorBuffer - 1 Agent only here'''
#             #     states, actions, rewards, next_states, dones = buffer.sample(agent_id=agent.id, device=self.device)
#             #     dones = dones.bool()
#             # except:
#             #     states, actions, rewards, next_states, dones = buffer.sample(device=self.device)
#             #     rewards = rewards.view(-1, 1)
#             #     dones = dones.byte()
#
#             # print('from buffer Obs {} Acs {} Rew {} NextObs {} Dones {}:'.format(states[:3], actions[:3], rewards[:3], next_states[:3], dones[:3]))
#             # # print('from buffer Acs: {} \n'.format(actions))
#             agent.optimizer.zero_grad()
#             # 1) GRAB Q_VALUE(s_j, a_j)
#             input_v = torch.cat([states, actions], dim=-1)
#             state_action_values = agent.policy(input_v)
#
#             # 2) GRAB MAX[Q_HAT_VALUES(s_j+1)]
#             # For the observations s_j+1, select an action using the Policy and calculate Q values of those using the Target net
#             target_next_state_actions = torch.tensor([agent.get_action_target(s_i) for s_i in next_states]).to(self.device)
#             input_v = torch.cat([next_states, target_next_state_actions], dim=-1).float()
#             next_state_values = agent.target_policy(input_v)
#
#             # 3) Overwrite 0 on all next_state_values where they were termination states
#             next_state_values[dones] = 0.0
#
#             # 4) Detach magic
#             # We detach the value from its computation graph to prevent gradients from flowing into the neural network used to calculate Q approximation next states.
#             # Without this our backpropagation of the loss will start to affect both predictions for the current state and the next state.
#             next_state_values = next_state_values.detach()
#             expected_state_action_values = next_state_values * self.gamma + rewards
#
#             # print(done_mask.shape, next_state_values.shape)
#             # print(expected_state_action_values.shape, rewards_v.shape)
#             self.loss = self.loss_calc(state_action_values, expected_state_action_values)
#
#             # The only issue is referencing the learner from here for the first parameter
#             # shiva.add_summary_writer(, agent, 'Loss per Step', loss_v, step_n)
#
#             self.loss.backward()
#             agent.optimizer.step()
#
#             if step_n % self.c == 0:
#                 # print('update target')
#                 agent.target_policy.load_state_dict(agent.policy.state_dict()) # Assuming is PyTorch!
# >>>>>>> robocup-pbt-mpi

    # def update_using_SimpleBuffer(self, agent, buffer, step_n, episodic=False):
    #     '''
    #         Implementation
    #             1) Calculate what the current expected Q val from each sample on the replay buffer would be
    #             2) Calculate loss between current and past reward
    #             3) Optimize
    #             4) If agent steps reached C, update agent.target network
    #
    #         Input
    #             agent       Agent reference who we are updating
    #             buffer      Buffer reference to sample from
    #             step_n      Current step number
    #             episodic    Flag indicating if update is episodic
    #     '''
    #     if episodic:
    #         '''
    #             DQN updates at every timestep, here we avoid doing an extra update after the episode terminates
    #         '''
    #         return
    #
    #     states, actions, rewards, next_states, dones = buffer.sample(device=self.device)
    #
    #     # # make tensors as needed
    #     # states_v = torch.tensor(states).float().to(self.device)
    #     # next_states_v = torch.tensor(next_states).float().to(self.device)
    #     # actions_v = torch.tensor(actions).to(self.device)
    #     # rewards_v = torch.tensor(rewards).to(self.device).view(-1, 1)
    #     # done_mask = torch.tensor(dones, dtype=torch.bool).to(self.device)
    #
    #     # print('from buffer:', states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape, '\n')
    #
    #     agent.optimizer.zero_grad()
    #     # 1) GRAB Q_VALUE(s_j, a_j) from minibatch
    #     # input_v = torch.tensor([ np.concatenate([s_i, a_i]) for s_i, a_i in zip(states, actions) ]).float().to(self.device)
    #
    #     state_action_values = agent.policy(input_v)
    #
    #     state_action_values = agent.policy(input_v)
    #     # 2) GRAB MAX[Q_HAT_VALUES(s_j+1)]
    #     # For the observations s_j+1, select an action using the Policy and calculate Q values of those using the Target net
    #
    #     input_v = torch.tensor([np.concatenate([s_i, agent.find_best_action(agent.target_policy, s_i)]) for s_i in
    #                             next_states]).float().to(self.device)
    #
    #     next_state_values = agent.target_policy(input_v)
    #     # 3) Overwrite 0 on all next_state_values where they were termination states
    #     next_state_values[done_mask] = 0.0
    #     # 4) Detach magic
    #     # We detach the value from its computation graph to prevent gradients from flowing into the neural network used to calculate Q approximation next states.
    #     # Without this our backpropagation of the loss will start to affect both predictions for the current state and the next state.
    #     next_state_values = next_state_values.detach()
    #     expected_state_action_values = next_state_values * self.gamma + rewards_v
    #
    #     # print(done_mask.shape, next_state_values.shape)
    #     # print(expected_state_action_values.shape, rewards_v.shape)
    #     self.loss = self.loss_calc(state_action_values, expected_state_action_values)
    #
    #     # The only issue is referencing the learner from here for the first parameter
    #     # shiva.add_summary_writer(, agent, 'Loss per Step', loss_v, step_n)
    #
    #     self.loss.backward()
    #     agent.optimizer.step()
    #
    #     if step_n % self.c == 0:
    #         agent.target_policy.load_state_dict(agent.policy.state_dict())  # Assuming is PyTorch!

    def get_loss(self):
        return self.loss

    def create_agent_of_role(self, id, role):
        self.role = role
        self.configs['Agent']['role'] = role
        self.agent = DQNAgent(id, self.obs_space[role], sum(self.acs_space['acs_space']), self.configs)
        return self.agent

    def get_metrics(self, episodic=False, agent_id=0):
        return self._metrics[agent_id] if agent_id in self._metrics else []

    def __str__(self):
        return '<DQNAlgorithm(num_updates={})>'.format(self.num_updates)
