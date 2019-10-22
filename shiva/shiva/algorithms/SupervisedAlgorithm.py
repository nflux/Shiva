import numpy as np
import torch
import helpers.misc as misc
from agents.ImitationAgent import ImitationAgent
from .Algorithm import Algorithm

class SupervisedAlgorithm(Algorithm):
    def __init__(self,
        observation_space: int,
        action_space: int,
        loss_function: object,
        regularizer: object,
        recurrence: bool,
        optimizer: object,
        gamma: np.float,
        learning_rate: np.float,
        beta: np.float,
        epsilon: set(),
        C: int,
        configs: dict):

        super(SupervisedAlgorithm, self).__init__(observation_space, action_space, loss_function, regularizer, recurrence, optimizer, gamma, learning_rate, beta, configs)
        self.C = C
        self.totalLoss = 0
        self.loss = 0
        self.loss_calc = self.loss_function()


    def update(self, agent, minibatch, step_n):
        '''
            Implementation
                1) Collect trajectories from the expert agent on a replay buffer
                2) Calculate the Cross Entropy Loss between imitation agent and
                    expert agent actions
                3) Optimize

            Input
                agent        Agent who we are updating
                expert agent Agent from which we are imitating
                minibatch    Batch from the experience replay buffer

            Returns
                None
        '''

        states, actions, rewards, next_states, dones = minibatch

        rewards_v = torch.tensor(rewards).to(self.device)
        done_mask = torch.ByteTensor(dones).to(self.device)

        # zero optimizer
        agent.optimizer.zero_grad()


        imitation_input_v = torch.tensor(states).float().to(self.device)
        #the output of the imitation agent is a probability distribution across all possible actions
        action_prob_dist = agent.policy(imitation_input_v)
        #Cross Entropy takes in action class as target value
        actions = torch.LongTensor(np.argmax(actions,axis = 1))
        actions = actions.detach()

        #action_prob_dist[done_mask] = 0.0


        #next_state_values[done_mask] = 0.0
        # 4) Detach magic
        # We detach the value from its computation graph to prevent
        # gradients from flowing into the neural network used to calculate Q
        # approximation for next states.
        # Without this our backpropagation of the loss will start to affect both
        # predictions for the current state and the next state.
        #action_prob_dist = action_prob_dist.detach()

        #We are using cross entropy loss between the action our imitation Policy
        #would choose, and the actions the expert agent took

        loss_v = self.loss_calc(action_prob_dist, actions).to(self.device)


        self.totalLoss += loss_v
        self.loss = loss_v

        loss_v.backward()
        agent.optimizer.step()


    def get_action(self, agent, observation) -> np.ndarray:

        best_act = self.find_best_action(agent.policy, observation)

        return best_act # replay buffer store lists and env does np.argmax(action)

    def find_best_action(self, network, observation: np.ndarray) -> np.ndarray:

        obs_v = torch.tensor(observation).float().to(self.device)
        best_q, best_act_v = float('-inf'), torch.zeros(self.action_space).to(self.device)
        for i in range(self.action_space):
            act_v = misc.action2one_hot_v(i)
            q_val = network(torch.cat([obs_v, act_v.to(self.device)]))
            if q_val > best_q:
                best_q = q_val
                best_act_v = act_v
        best_act = best_act_v.tolist()
        return best_act

    def create_agent(self, id):
        new_agent = ImitationAgent(id, self.observation_space, self.action_space, self.optimizer_function, self.learning_rate, self.configs)
        self.agents.append(new_agent)
        return new_agent

    def get_loss(self):
        return self.loss

    def get_average_loss(self, step):
        average = self.totalLoss/step
        self.totalLoss = 0
        return average