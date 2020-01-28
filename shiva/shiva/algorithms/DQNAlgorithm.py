import numpy as np
import torch
import random

from shiva.algorithms.Algorithm import Algorithm
from shiva.agents.DQNAgent import DQNAgent
from shiva.helpers.misc import action2one_hot

class DQNAlgorithm(Algorithm):
    def __init__(self, obs_space, acs_space, configs):
        '''
            Inputs
                epsilon        (start, end, decay rate), example: (1, 0.02, 10**5)
                C              Number of iterations before the target network is updated
        '''
        super(DQNAlgorithm, self).__init__(obs_space, acs_space, configs)
        torch.manual_seed(self.manual_seed)
        np.random.seed(self.manual_seed)
        self.acs_space = acs_space['acs_space']
        self.obs_space = obs_space
        self.loss = 0

    def update(self, agent, buffer, step_n, episodic=False):
        '''
            Implementation
                1) Calculate what the current expected Q val from each sample on the replay buffer would be
                2) Calculate loss between current and past reward
                3) Optimize
                4) If agent steps reached C, update agent.target network

            Input
                agent       Agent reference who we are updating
                buffer      Buffer reference to sample from
                step_n      Current step number or done_count when doing episodic updates!!!!
                episodic    Flag indicating if update is episodic
        '''
        # if episodic: # for DQN to do step-wise updates only
        if not episodic: # for DQN to do episodic update
            return

        try:
            '''For MultiAgentTensorBuffer - 1 Agent only here'''
            states, actions, rewards, next_states, dones = buffer.sample(agent_id=0, device=self.device)
            dones = dones.bool()
        except:
            states, actions, rewards, next_states, dones = buffer.sample(device=self.device)
            rewards = rewards.view(-1, 1)

        # print('from buffer:', states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape, '\n')

        agent.optimizer.zero_grad()
        # 1) GRAB Q_VALUE(s_j, a_j)
        input_v = torch.cat([states, actions], dim=-1)
        state_action_values = agent.policy(input_v)

        # 2) GRAB MAX[Q_HAT_VALUES(s_j+1)]
        # For the observations s_j+1, select an action using the Policy and calculate Q values of those using the Target net
        target_next_state_actions = torch.tensor([agent.get_action_target(s_i) for s_i in next_states]).to(self.device)
        input_v = torch.cat([next_states, target_next_state_actions], dim=-1)
        next_state_values = agent.target_policy(input_v)

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
        agent.optimizer.step()

        if step_n % self.c == 0:
            # print('update target')
            agent.target_policy.load_state_dict(agent.policy.state_dict()) # Assuming is PyTorch!

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

    def get_action(self, agent, observation, step_n) -> np.ndarray:
        assert "DoNotUse"
        '''
            With the probability epsilon we take the random action,
            otherwise we use the network to obtain the best Q-value per each action
        '''
        # if step_n < self.exploration_steps:
        #     action_idx = random.sample(range(self.acs_space), 1)[0]
        #     action = action2one_hot(self.acs_space, action_idx, numpy=False)
        # elif random.uniform(0, 1) < max(self.epsilon_end, self.epsilon_start - (step_n / self.epsilon_decay)):
        #     # this might not be correct implementation of e greedy
        #     action_idx = random.sample(range(self.acs_space), 1)[0]
        #     action = action2one_hot(self.acs_space, action_idx)
        # else:
        #     if len(observation.shape) > 1:
        #         # print('here')
        #         pass
        #     # Iterate over all the actions to find the highest Q value
        #     action = agent.get_action(observation)
        # return action # replay buffer store lists and env does np.argmax(action)

    def get_loss(self):
        return self.loss

    def create_agent(self, agent_id):
        self.agent = DQNAgent(agent_id, self.obs_space, self.acs_space, self.configs['Agent'], self.configs['Network'])
        return self.agent

    def get_metrics(self, episodic=False):
        if not episodic:
            metrics = [
                ('Algorithm/Loss_per_Step', self.loss),
                ('Agent/learning_rate', self.agent.learning_rate)
            ]
        else:
            metrics = []
        return metrics

    def __str__(self):
        return 'DQNAlgorithm'