'''
TODO
    - DQAlgorithm._is_epsilon_greedy_action() function to be a decorator so that can be used for other algorithms
'''

def initialize_algorithm(observation_space: int, action_space: int, _params: list):
    
    if _params[0]['algorithm'] == 'DQN':
        return DQAlgorithm(
            observation_space = observation_space,
            action_space = action_space,
            loss_function = getattr(torch.nn, _params[0]['loss_function']), 
            regularizer = _params[0]['regularizer'],
            recurrence = _params[0]['recurrence'], 
            optimizer = getattr(torch.optim, _params[0]['optimizer']), 
            gamma = float(_params[0]['gamma']), 
            learning_rate = float(_params[0]['learning_rate']),
            beta = _params[0]['beta'],
            epsilon = (float(_params[0]['epsilon_start']), float(_params[0]['epsilon_end']), float(_params[0]['epsilon_decay'])),
            C = _params[0]['c'],
            configs = {
                'agent': _params[1],
                'network': _params[2]
            }
        )
    else:
        return None

import random
import numpy as np

import torch

import Agent
import utils.Noise

class AbstractAlgorithm():
    def __init__(self,
        observation_space: np.ndarray,
        action_space: np.ndarray,
        loss_function: object, 
        regularizer: object, 
        recurrence: bool, 
        optimizer_function: object, 
        gamma: np.float, 
        learning_rate: np.float,
        beta: np.float,
        configs: dict
        ):
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
        self.observation_space = observation_space
        self.action_space = action_space
        self.loss_function = loss_function
        self.regularizer = regularizer
        self.recurrence = recurrence
        self.optimizer_function = optimizer_function
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.beta = beta
        self.configs = configs


        self.loss_calc = self.loss_function()
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

    def get_agents(self):
        return self.agents

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
        learning_rate: np.float,
        beta: np.float,
        epsilon: set(),
        C: int,
        configs: dict):
        '''
            Inputs
                epsilon        (start, end, decay rate), example: (1, 0.02, 10**5)
                C              Number of iterations before the target network is updated
        '''
        super(DQAlgorithm, self).__init__(observation_space, action_space, loss_function, regularizer, recurrence, optimizer, gamma, learning_rate, beta, configs)
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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        states, actions, rewards, next_states, dones = minibatch
        # make tensors
        # states_v = torch.tensor(states).float()#.to(device)
        # next_states_v = torch.tensor(next_states).float()#.to(device)
        # actions_v = torch.tensor(actions)#.to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.ByteTensor(dones).to(device)

        # zero optimizer
        agent.optimizer.zero_grad()
        # 1) GRAB Q_VALUE(s_j, a_j)
        # We pass observations to the first model and extract the
        # specific Q-values for the taken actions using the gather() tensor operation.
        # The first argument to the gather() call is a dimension index that we want to
        # perform gathering on (equal to 1, which corresponds to actions). 
        # The second argument is a tensor of indices of elements to be chosen
        input_v = torch.tensor([ np.concatenate([s_i, a_i]) for s_i, a_i in zip(states, actions) ]).float().to(device)
        state_action_values = agent.policy(input_v) #.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        # 2) GRAB MAX[Q_HAT_VALUES(s_j+1)]
        # We apply the target network to our next state observations and 
        # calculate the maximum Q-value along the same action dimension 1.
        # Function max() returns both maximum values and indices of those values (so it calculates both max and argmax), 
        # which is very convenient. However, in this case, we’re interested only in values, so we take
        # the first entry of the result.
        input_v = torch.tensor([ np.concatenate([s_i, self.find_best_action(agent.target_policy, s_i)]) for s_i in next_states ]).float().to(device)
        next_state_values = agent.target_policy(input_v)#.max(1)[0]
        # 3) OVERWRITE 0 ON ALL Q_HAT_VALUES WHERE s_j IS A TERMINATION STATE
        # If transition in the batch is from the last step in the episode, then our value of the action doesn’t have a
        # discounted reward of the next state, as there is no next state to gather reward from
        next_state_values[done_mask] = 0.0
        # 4) Detach magic
        # We detach the value from its computation graph to prevent
        # gradients from flowing into the neural network used to calculate Q
        # approximation for next states.
        # Without this our backpropagation of the loss will start to affect both 
        # predictions for the current state and the next state.
        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + rewards_v

        loss_v = self.loss_calc(state_action_values, expected_state_action_values)
        loss_v.backward()
        agent.optimizer.step()

        if step_n % self.C == 0:
            agent.target_policy.load_state_dict(agent.policy.state_dict()) # Assuming is PyTorch!

    '''
        With the probability epsilon we take the random action,
        otherwise we use the network to obtain the best Q-value per each action
    '''
    def get_action(self, agent, observation, step_n) -> np.ndarray:
        epsilon = max(self.epsilon_end, self.epsilon_start - (step_n / self.epsilon_decay))
        if random.uniform(0, 1) < epsilon:
            action = random.sample(range(self.action_space), 1)[0]
            best_act = self.action2one_hot(action)
        else:
            best_act = self.find_best_action(agent.policy, observation)
        return best_act # replay buffer store lists and env does np.argmax(action)

    '''
        Iterates over the action space and returns a one-hot encoded list
    '''
    def find_best_action(self, network, observation: np.ndarray) -> np.ndarray:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        obs_v = torch.tensor(observation).float().to(device)
        best_q, best_act_v = float('-inf'), torch.zeros(self.action_space).to(device)
        for i in range(self.action_space):
            act_v = self.action2one_hot_v(i)
            q_val = network(torch.cat([obs_v, act_v.to(device)]))
            if q_val > best_q:
                best_q = q_val
                best_act_v = act_v
        best_act = best_act_v.tolist()
        return best_act

    def action2one_hot(self, action_idx: int) -> np.ndarray:
        z = np.zeros(self.action_space)
        z[action_idx] = 1
        return z

    def action2one_hot_v(self, action_idx: int) -> torch.tensor:
        z = torch.zeros(self.action_space)
        z[action_idx] = 1
        return z

    def create_agent(self):
        new_agent = Agent.DQAgent(self.observation_space, self.action_space, self.optimizer_function, self.learning_rate, self.configs)
        self.agents.append(new_agent)
        return new_agent

##########################################################################
#    
#    DDPG Algorithm Implementation
#    
##########################################################################

class DDPGAlgorithm(AbstractAlgorithm):
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
        '''
            Inputs
                epsilon        (start, end, decay rate), example: (1, 0.02, 10**5)
                C              Number of iterations before the target network is updated
        '''
        super(DQAlgorithm, self).__init__(observation_space, action_space, loss_function, regularizer, recurrence, optimizer, gamma, learning_rate, beta, configs)
        self.epsilon_start = epsilon[0]
        self.epsilon_end = epsilon[1]
        self.epsilon_decay = epsilon[2]
        self.C = C

        self.ou_noise = Noise.OUNoise(self.action_space)

    def update(self, agent, minibatch, step_n):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        states, actions, rewards, dones, next_states = minibatch
        
        states = torch.tensor(states).to(device)
        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards).to(device)
        dones_mask = torch.ByteTensor(dones).to(device)
        next_states = torch.tensor(next_states).to(device)

        # train critic
        agent.critic_optimizer.zero_grad()
        next_state_actions = agent.target_actor(next_states)
        q_next_states = agent.target_critic(next_states, next_state_actions)
        q_next_states[dones_mask] = 0.0
        y_i = rewards + self.gamma * q_next_states
        
        q_now = agent.critic(states, actions)

        critic_loss = self.loss_calc(q_now, y_i.detach())
        critic_loss.backward()
        agent.critic_optimizer.step()

        # train actor
        agent.actor_optimizer.zero_grad()
        

    def get_action(self, agent, observation, step_n) -> np.ndarray: # maybe a torch.tensor
        return agent.actor(observation).data.numpy() + self.ou_noise.noise()

    def create_agent(self):
        new_agent = Agent.DDPGAgent(self.observation_space, self.action_space, self.optimizer_function, self.learning_rate, self.configs)
        self.agents.append(new_agent)
        return new_agent