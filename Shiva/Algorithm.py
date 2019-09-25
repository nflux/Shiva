'''
TODO
    - Init function
    - DQAlgorithm._is_epsilon_greedy_action() function to be a decorator so that can be used for other algorithms


'''

def init_algorithm(config, obs_dim, ):
    _return = None
    if config.algorithm == 'DQN':
        
    else:

    return _return

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
                loss_function:      Function used to calculate the loss during training
                regularizer:        ?
                recurrence:         ?
                optimizer:          Optimization algorithm to train network weights
                gamma:              Hyperparameter
                batch_size:         ?
                learning_rate       Learning rate used in the optimizer
                beta:               Hyperparameter
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
        self.n_steps = 0

    # def __repr__(self):
    #     return "AbstractAlgorithm(\n\tObs_space:{0},\n\tAct_space:{1},\n\tLossFunction:{2},\n\tOptimizer:{3},\n\tLearningRate:{4},\n\tRegularizer:{5},\n\tRecurrence:{6},\n\tGamma:{7},\n\tBeta:{8},\n\tBatch_size:{9}\n)".format(
    #         self.observation_space,
    #         self.action_space,
    #         self.loss_function,
    #         self.optimizer,
    #         self.learning_rate
    #         self.regularizer,
    #         self.recurrence,
    #         self.gamma,
    #         self.beta,
    #         self.batch_size
    #     )

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
##########################################################################

'''
    DQ Algorithm Implementation
    
    Discrete Action Space
'''

class DQAlgorithm(AbstractAlgorithm):
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
        beta: np.float,
        g_epsilon: np.float):
        super(DQN, self).__init__(observation_space, action_space, loss_function, regularizer, recurrence, optimizer, gamma, batch_size, learning_rate, beta)
        self.g_epsilon = g_epsilon

    def update(self, agent, data):
        

    def get_action(self, agent, observation):
        '''
            Perform an Epsilon-Greedy action
        '''
        if self._is_epsilon_greedy_action():
            random.sample(range(agent.obs_dim), 1)
        else:
            agent.policy(observation)

    def _is_epsilon_greedy_action(self, action):
        '''
            - This function should be implemented as a Decorator
            - Do we want Epsilon to decay????
        '''
        return random.uniform(0, 1) < self.g_epsilon:

    def create_agent(self):
        new_agent = DQAgent(self.observation_space.shape[0], self.action_space.shape[0])
        self.agents.append(new_agent)
        return new_agent