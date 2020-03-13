import numpy as np
import torch
import random

from shiva.algorithms.Algorithm import Algorithm
from shiva.agents.IRLAgent import IRLAgent
from shiva.helpers.misc import action2one_hot
from shiva.core.admin import Admin
from shiva.networks import DynamicLinearNetwork, SupervisedNeuralNetwork
# from shiva.helpers.config_handler import load_class // maybe not needed


class IRLAlgorithm(Algorithm):
    '''

        Experimental Algorithm for Inverse Reinforcement Learning

            - Will sample from a the buffer
            - Will use various methods of calculating loss
            - Something experimental could be providing the actual rewards it gave, updating in a DQN fashion with a
                target network.
            - Accepted approach is that losses are calculated using supervised learning algorithms
            - Currently supervised methods being considered are the following:
                - Logistic Regression
                - Support Vector Machines
                - Artificial Neural Network (Supervised Style)  // This will be tried first

                - The supervised algorithms might be imported from other files to be reusable and make this code more
                    readable.


        This will host the reward neural network

        One thing is for certain I need to map a state action pair to a reward.
            - If the network is giving a state action pair and produces a reward then how would I know what
            the true reward is supposed to be. How will I update the network?

            - How will a supervised learning algorithm know whether or not

        It will have to save the network after every update and will have to be protected by a flag or something.

        This will also utilize the supervised learning algorithms to make predictions.
            - Score predictions?
            - Preference predictions?

        I know we'll use the predictions, whether they are by the neural network or supervised learning algorithms
        to calculate the loss using Cross Entropy.
        I think either could be updated using that loss.

        Bradley-Terry model will be used for estimating score functions from pairwise preferences.

        The estimated reward is defined by independently normalizing each of these predictors and averaging
        the results.

        1/e of data must be held out for validation.

    '''

    def __init__(self, obs_space, acs_space, configs):
        '''
            Inputs
                epsilon        (start, end, decay rate), example: (1, 0.02, 10**5)
                C              Number of iterations before the target network is updated
        '''
        super(IRLAlgorithm, self).__init__(obs_space, acs_space, configs)
        self.configs = configs
        torch.manual_seed(self.manual_seed)
        np.random.seed(self.manual_seed)
        self.acs_space = acs_space['acs_space']
        self.obs_space = obs_space
        self.state_action_space = self.acs_space + self.obs_space
        self.loss = 0
        self.expert = Admin._load_agent(self.expert_path)

        # self.expert_predictor = SupervisedNeuralNetwork(self.state_action_space, 1, configs[2]['expert_predictor'])

    def update(self, agent, buffer, step_n, episodic=False):

        if episodic:
            '''
                IRL updates at every step, here we avoid doing an extra update after the episode terminates
            '''
            return

        states, actions, _, _, _, _ = buffer.full_buffer(device=self.device)

        # print('from buffer:', states.shape, actions.shape '\n')
        # input()

        self.loss = 0
        agent.optimizer.zero_grad()

        for state, action in zip(states, actions):
            expert_action = torch.tensor(self.expert.agent.get_action(state))
            # expert_reward = agent.get_reward(state, expert_action)
            expert_reward = torch.tensor([5.0])
            agent_reward = agent.get_reward(state, action)

            # self.loss += -(torch.exp(expert_reward) / torch.exp(agent_reward))

            if torch.all(torch.eq(action, expert_action)):
                """ Expert Action and Agent Action were the same """
                self.loss += -(torch.exp(expert_reward) / torch.exp(agent_reward))
            else:
                """ Expert Action and Agent Action were different """
                self.loss += (torch.exp(expert_reward) / torch.exp(agent_reward))

        agent.optimizer.step()
        self.loss.backward()

    def get_heuristic_loss(self, states, actions):

        """

            This function will use the expert to identify whether actions were expert actions given the state

        """

        loss = 0

        for i in range(len(actions)):
            # this seems suspicious, why am I calling expert.agent.get_action?
            # I think it should be self.expert.get_action(states[i])
            expert_action = torch.tensor(self.expert.agent.get_action(states[i]))
            loss += abs(torch.argmax(actions[i]) - max(expert_action))
        return loss

    def create_agent(self):
        return IRLAgent(self.id_generator(), self.obs_space, self.acs_space, self.configs[1], self.configs[2])

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
        return 'IRLAlgorithm'