import numpy as np
import torch

class Algorithm():
    def __init__(self, obs_space, acs_space, configs):
        '''
            Input
                observation_space   Shape of the observation space, aka input to policy network
                action_space        Shape of the action space, aka output from policy network
                loss_function       Function used to calculate the loss during training
                regularizer
                recurrence
                optimizer           Optimization function to train network weights
                gamma               Hyperparameter
                learning_rate       Learning rate used in the optimizer
                beta                Hyperparameter
        '''
        self.configs = configs
        {setattr(self, k, v) for k, v in self.configs['Algorithm'].items()}
        self.agentCount = 0
        self.agents = []
        self.observation_space = obs_space
        self.action_space = acs_space
        self.loss_calc = getattr(torch.nn, self.configs['Algorithm']['loss_function'])()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = np.random.randint(0, 100)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)


    def update(self, agent, data, episodic=False):
        '''
            Updates the agents network using the data

            Input
                agent:      the agent who we want to update it's network
                data:       data used to train the network
                episodic:   flag indicating if the update is episodic or per timestep

            Return
                None
        '''
        assert "Method Not Implemented"

    def get_action(self, agent, observation):
        '''
            Determines the best action for the agent and a given observation

            Input
                agent:          the agent we want the action
                observation:

            Return
                Action
        '''
        assert "Method Not Implemented"

    def create_agent(self):
        '''
            Creates a new agent

            Input

            Return
                Agent
        '''
        assert "Method Not Implemented"

    def id_generator(self):
        agent_id = self.agentCount
        self.agentCount += 1
        return agent_id

    def get_agents(self):
        return self.agents
