import numpy as np
import torch
import random
from agents.DQNAgent import DQNAgent
import helpers.misc as misc
from .Algorithm import Algorithm

class DQNAlgorithm(Algorithm):
    def __init__(self, config):
        '''
            Inputs
                epsilon        (start, end, decay rate), example: (1, 0.02, 10**5)
                C              Number of iterations before the target network is updated
        '''
        super(DQNAlgorithm, self).__init__(config)
        self.totalLoss = 0
        self.loss = 0
        self.agentCount = 0

    def update(self, agent, minibatch, step_n):
        '''
            Implementation
                1) Calculate what the current expected Q val from each sample on the replay buffer would be
                2) Calculate loss between current and past reward
                3) Optimize
                4) If agent steps reached C, update agent.target network

            Input
                agent       Agent who we are updating
                minibatch   Batch from the experience replay buffer

            Returns
                None
        '''

        states, actions, rewards, next_states, dones = minibatch
        # make tensors as needed
        # states_v = torch.tensor(states).float().to(self.device)
        # next_states_v = torch.tensor(next_states).float().to(self.device)
        # actions_v = torch.tensor(actions).to(self.device)
        rewards_v = torch.tensor(rewards).to(self.device)
        done_mask = torch.tensor(dones, dtype=torch.bool).to(self.device)

        agent.optimizer.zero_grad()
        # 1) GRAB Q_VALUE(s_j, a_j) from minibatch
        input_v = torch.tensor([ np.concatenate([s_i, a_i]) for s_i, a_i in zip(states, actions) ]).float().to(self.device)
        state_action_values = agent.policy(input_v)
        # 2) GRAB MAX[Q_HAT_VALUES(s_j+1)]
        # For the observations s_j+1, select an action using the Policy and calculate Q values of those using the Target net
        input_v = torch.tensor([ np.concatenate([s_i, agent.get_action_target(s_i) ]) for s_i in next_states ]).float().to(self.device)
        next_state_values = agent.target_policy(input_v)
        # 3) Overwrite 0 on all next_state_values where they were termination states
        next_state_values[done_mask] = 0.0
        # 4) Detach magic
        # We detach the value from its computation graph to prevent gradients from flowing into the neural network used to calculate Q approximation next states.
        # Without this our backpropagation of the loss will start to affect both predictions for the current state and the next state.
        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + rewards_v

        loss_v = self.loss_calc(state_action_values, expected_state_action_values)

        self.totalLoss += loss_v
        self.loss = loss_v

        loss_v.backward()
        agent.optimizer.step()

        if step_n % self.c == 0:
            agent.target_policy.load_state_dict(agent.policy.state_dict()) # Assuming is PyTorch!
    
    def get_action(self, agent, observation, step_n) -> np.ndarray:
        '''
            With the probability epsilon we take the random action,
            otherwise we use the network to obtain the best Q-value per each action
        '''
        epsilon = max(self.epsilon_end, self.epsilon_start - (step_n / self.epsilon_decay))
        if random.uniform(0, 1) < epsilon:
            action_idx = random.sample(range(self.action_space), 1)[0]
            action = misc.action2one_hot(self.action_space, action_idx)
        else:
            # Iterate over all the actions to find the highest Q value
            action = agent.get_action(observation)
        return action # replay buffer store lists and env does np.argmax(action)

    def get_loss(self):
        return self.loss

    def get_average_loss(self, step):
        average = self.totalLoss/step
        self.totalLoss = 0
        return average

    def create_agent(self, agent_config, net_config):
        self.agent = DQNAgent(self.id_generator(), agent_config, net_config)
        return self.agent

    def id_generator(self):
        agent_id = self.agentCount
        self.agentCount +=1
        return agent_id