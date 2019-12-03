import numpy as np
import torch
import random
from agents.DQNAgent import DQNAgent
import helpers.misc as misc
from .Algorithm import Algorithm
from settings import shiva

class DQNAlgorithm(Algorithm):
    def __init__(self, obs_space, acs_space, configs):
        '''
            Inputs
                epsilon        (start, end, decay rate), example: (1, 0.02, 10**5)
                C              Number of iterations before the target network is updated
        '''
        super(DQNAlgorithm, self).__init__(obs_space, acs_space, configs)
        self.acs_space = acs_space
        self.obs_space = obs_space
        self.loss = 0

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
        states_v = torch.tensor(states).float().to(self.device)
        next_states_v = torch.tensor(next_states).float().to(self.device)
        actions_v = torch.tensor(actions).to(self.device)
        rewards_v = torch.tensor(rewards).to(self.device).view(-1, 1)
        done_mask = torch.tensor(dones, dtype=np.bool).to(self.device)

        print('from buffer:', states_v.shape, actions_v.shape, rewards_v.shape, next_states_v.shape, done_mask.shape, '\n')
        # input()


        agent.optimizer.zero_grad()
        # 1) GRAB Q_VALUE(s_j, a_j) from minibatch
        input_v = torch.tensor([ np.concatenate([s_i, a_i]) for s_i, a_i in zip(states, actions) ]).float().to(self.device)
        print(input_v.shape)
        input()
        state_action_values = agent.policy(input_v)
        # 2) GRAB MAX[Q_HAT_VALUES(s_j+1)]
        # For the observations s_j+1, select an action using the Policy and calculate Q values of those using the Target net

        input_v = torch.tensor([np.concatenate( [s_i, agent.find_best_action(agent.target_policy, s_i )]) for s_i in next_states ] ).float().to(self.device)

        next_state_values = agent.target_policy(input_v)
        # 3) Overwrite 0 on all next_state_values where they were termination states
        next_state_values[done_mask] = 0.0
        # 4) Detach magic
        # We detach the value from its computation graph to prevent gradients from flowing into the neural network used to calculate Q approximation next states.
        # Without this our backpropagation of the loss will start to affect both predictions for the current state and the next state.
        next_state_values = next_state_values.detach()
        expected_state_action_values = next_state_values * self.gamma + rewards_v

        # print(done_mask.shape, next_state_values.shape)
        # print(expected_state_action_values.shape, rewards_v.shape)
        self.loss = self.loss_calc(state_action_values, expected_state_action_values)

        # The only issue is referencing the learner from here for the first parameter
        # shiva.add_summary_writer(, agent, 'Loss per Step', loss_v, step_n)

        self.loss.backward()
        agent.optimizer.step()

        if step_n % self.c == 0:
            agent.target_policy.load_state_dict(agent.policy.state_dict()) # Assuming is PyTorch!
    
    def get_action(self, agent, observation, step_n) -> np.ndarray:
        '''
            With the probability epsilon we take the random action,
            otherwise we use the network to obtain the best Q-value per each action
        '''
        if step_n < self.exploration_steps:
            action_idx = random.sample(range(self.acs_space), 1)[0]
            action = misc.action2one_hot(self.acs_space, action_idx, numpy=False)
        elif random.uniform(0, 1) < max(self.epsilon_end, self.epsilon_start - (step_n / self.epsilon_decay)):
            # this might not be correct implementation of e greedy
            print('greedy')
            action_idx = random.sample(range(self.acs_space), 1)[0]
            action = misc.action2one_hot(self.acs_space, action_idx)
        else:
            # Iterate over all the actions to find the highest Q value
            action = agent.get_action(observation)
        return action # replay buffer store lists and env does np.argmax(action)

    def get_loss(self):
        return self.loss

    def create_agent(self, id):
        print(self.configs[1])
        self.agent = DQNAgent(id, self.obs_space, self.acs_space, self.configs[1], self.configs[2])
        return self.agent
