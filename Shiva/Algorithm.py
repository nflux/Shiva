'''
TODO
    - Init function
    - 

    - DQAlgorithm._is_epsilon_greedy_action() function to be a decorator so that can be used for other algorithms


'''

import Agent

import random
import numpy as np

class AbstractAlgorithm():
    def __init__(self,
        observation_space: np.ndarray,
        action_space: np.ndarray,
        loss_function: object, 
        regularizer: object, 
        recurrence: bool, 
        optimizer: object, 
        gamma: np.float, 
        batch_size: int, 
        learning_rate: np.float,
        beta: np.float):
        '''
            Input
                observation_space   Shape of the observation space, aka input to policy network
                action_space        Shape of the action space, aka output from policy network
                loss_function       Function used to calculate the loss during training
                regularizer         
                recurrence          
                optimizer           Optimization function to train network weights
                gamma               Hyperparameter
                batch_size          
                learning_rate       Learning rate used in the optimizer
                beta                Hyperparameter
        '''
        self.observation_space = observation_space
        self.action_space = action_space
        self.loss_function = loss_function
        self.regularizer = regularizer
        self.recurrence = recurrence
        self.optimizer = optimizer
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta

        self.agents = []

    def update(self, agent, data):
        '''
            Updates the agents network using the data

            Input
                agent:  the agent who we want to update it's network
                data:   data used to train the network

            Return
                None
        '''
        pass

    def get_action(self, agent, observation):
        '''
            Determines the best action for the agent and a given observation

            Input
                agent:          the agent we want the action
                observation:    

            Return
                Action
        '''
        pass

    def create_agent(self):
        '''
            Creates a new agent

            Input

            Return
                Agent
        '''
        pass

##########################################################################
#    DQ Algorithm Implementation
#    
#    Discrete Action Space
##########################################################################

class DQAlgorithm(AbstractAlgorithm):
    def __init__(self,
        observation_space: int,
        action_space: int,
        loss_function: object, 
        regularizer: object, 
        recurrence: bool, 
        optimizer: object, 
        gamma: np.float, 
        batch_size: int, 
        learning_rate: np.float,
        beta: np.float,
        epsilon: set()=(1, 0.02, 10**5),
        C: int):
        '''
            Inputs
                epsilon        (start, end, decay rate)
                C              Number of iterations before the target network is updated
        '''
        super(DQN, self).__init__(observation_space, action_space, loss_function, regularizer, recurrence, optimizer, gamma, batch_size, learning_rate, beta)
        self.epsilon_start = epsilon[0]
        self.epsilon_end = epsilon[1]
        self.epsilon_decay = epsilon[2]
        self.C = C

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
        states, actions, rewards, dones, next_states = minibatch

        states_v = torch.tensor(states).to(device)
        next_states_v = torch.tensor(next_states).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.ByteTensor(dones).to(device)
        
        agent.optimizer.zero_grad()

        state_action_values = agent.policy(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_state_values = agent.target_policy(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * GAMMA + rewards_v

        loss_v = self.loss_function(state_action_values, expected_state_action_values)
        loss_v.backward()
        agent.optimizer.step()

        if step_n % self.C == 0:
            agent.target_policy.load_state_dict(agent.policy.state_dict()) # Assuming is PyTorch!

    def get_action(self, agent, observation, step_n):
        '''
            With the probability epsilon we take the random action,
            otherwise we use the past model to obtain the Q-values for all possible actions
            and choose the best
        '''
        epsilon = max(self.epsilon_end, self.epsilon_start - (step_n / self.epsilon_decay))
        if random.uniform(0, 1) < epsilon:
            action = random.sample(range(self.action_space), 1)
        else:
            obs_a = np.array([obs], copy=False)
            obs_v = torch.tensor(obs_a).to(device)
            q_vals_v = agent.policy(obs_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())
        return action

    def create_agent(self):
        new_agent = DQAgent(self.observation_space, self.action_space, self.optimizer, self.learning_rate)
        self.agents.append(new_agent)
        return new_agent