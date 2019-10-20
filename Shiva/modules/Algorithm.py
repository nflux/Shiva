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
    elif _params[0]['algorithm'] == 'Imitation':
        return SupervisedAlgorithm(
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
        ), DaggerAlgorithm(
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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



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

from Agent import DQAgent

# average loss per episode
# loss per step

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
        self.totalLoss = 0
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

        # make tensors
        states_v = torch.tensor(states).float().to(self.device)
        next_states_v = torch.tensor(next_states).float().to(self.device)
        actions_v = torch.tensor(actions).to(self.device)
        rewards_v = torch.tensor(rewards).to(self.device)
        done_mask = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # zero optimizer
        agent.optimizer.zero_grad()
        # 1) GRAB Q_VALUE(s_j, a_j)
        # We pass observations to the first model and extract the
        # specific Q-values for the taken actions using the gather() tensor operation.
        # The first argument to the gather() call is a dimension index that we want to
        # perform gathering on (equal to 1, which corresponds to actions).
        # The second argument is a tensor of indices of elements to be chosen
        input_v = torch.tensor([ np.concatenate([s_i, a_i]) for s_i, a_i in zip(states, actions) ]).float().to(self.device)
        state_action_values = agent.policy(input_v) #.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        # 2) GRAB MAX[Q_HAT_VALUES(s_j+1)]
        # We apply the target network to our next state observations and
        # calculate the maximum Q-value along the same action dimension 1.
        # Function max() returns both maximum values and indices of those values (so it calculates both max and argmax),
        # which is very convenient. However, in this case, we’re interested only in values, so we take
        # the first entry of the result.
        input_v = torch.tensor([ np.concatenate([s_i, self.find_best_action(agent.target_policy, s_i)]) for s_i in next_states ]).float().to(self.device)
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

        self.totalLoss += loss_v
        self.loss = loss_v

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
            # # Iterate over all the actions to find the highest Q value
            best_act = self.find_best_action(agent.policy, observation)
        return best_act # replay buffer store lists and env does np.argmax(action)

    '''
        Iterates over the action space and returns a one-hot encoded list
    '''
    def find_best_action(self, network, observation: np.ndarray) -> np.ndarray:

        obs_v = torch.tensor(observation).float().to(self.device)
        best_q, best_act_v = float('-inf'), torch.zeros(self.action_space).to(self.device)
        for i in range(self.action_space):
            act_v = self.action2one_hot_v(i)
            q_val = network(torch.cat([obs_v, act_v.to(self.device)]))
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

    # # Currently not being used.
    # def one_hot(self, tensor):
    #     _, act_idx = torch.max(tensor, dim=1) # dim=0 TBD! Works for now
    #     ret = torch.zeros(tensor[0].shape)
    #     ret[act_idx.item()] = 1
    #     return ret

    def get_loss(self):
        return self.loss

    def get_average_loss(self, step):
        average = self.totalLoss/step
        self.totalLoss = 0
        return average

    def create_agent(self, id):
        new_agent = DQAgent(id, self.observation_space, self.action_space, self.optimizer_function, self.learning_rate, self.configs)
        self.agents.append(new_agent)
        return new_agent

    def create_empty_agent(self):
        pass


##########################################################################
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

    def update(self, agent, minibatch, step_n):
        pass

    def create_agent(self, id):
        new_agent = DDPGAgent(id, self.observation_space, self.action_space, self.optimizer_function, self.learning_rate, self.configs)
        self.agents.append(new_agent)
        return new_agent


###############################################################################
#
#Supervised Algorithm Implementation
#
###############################################################################
from Agent import ImitationAgent

class SupervisedAlgorithm(AbstractAlgorithm):
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
            act_v = self.action2one_hot_v(i)
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


    def action2one_hot(self, action_idx: int) -> np.ndarray:
        z = np.zeros(self.action_space)
        z[action_idx] = 1
        return z

    def action2one_hot_v(self, action_idx: int) -> torch.tensor:
        z = torch.zeros(self.action_space)
        z[action_idx] = 1
        return z

    def one_hot2action(self,actions):
        #z = torch.zeros(self.action_space)
        for action in actions:
            action = np.argmax(action)
        return actions


    def get_loss(self):
        return self.loss

    def get_average_loss(self, step):
        average = self.totalLoss/step
        self.totalLoss = 0
        return average





###############################################################################
#
# Dagger Algorithm for Imitation Implementation
#
################################################################################

class DaggerAlgorithm(AbstractAlgorithm):
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

        super(DaggerAlgorithm, self).__init__(observation_space, action_space, loss_function, regularizer, recurrence, optimizer, gamma, learning_rate, beta, configs)
        self.C = C
        self.totalLoss = 0
        self.loss = 0

    def update(self, imitation_agent,expert_agent, minibatch, step_n):
        '''
            Implementation
                1) Collect Trajectories from the imitation policy. By choosing
                    actions according to our initial policy, we are allowing for
                    for further exploration, so we can encounter new observations
                    that the expert would not have visited in. This allows us to
                    encounter and learn from negative situations, as well as the
                    positive states the expert lead us through.
                2) Calculate the Cross Entropy Loss between the imitation policy's
                    actions and the actions the expert policy would have taken.
                3) Optimize

            Input
                agent       Agent who we are updating
                exper agent Agent we are imitating
                minibatch   Batch from the experience replay buffer

            Returns
                None
        '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        states, actions, rewards, next_states, dones = minibatch


        rewards_v = torch.tensor(rewards).to(self.device)
        done_mask = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # zero optimizer
        imitation_agent.optimizer.zero_grad()

        input_v = torch.tensor(states).float().to(self.device)
        actions_one_hot = torch.zeros((len(actions),2))
        action_prob_dist = imitation_agent.policy(input_v)


        expert_actions = torch.LongTensor(len(states))
        for i in range(len(states)):
            expert_actions[i] = self.find_best_expert_action(expert_agent.policy,states[i])
        expert_actions = expert_actions.detach()


        #Loss will be Cross Entropy Loss between the action probabilites produced
        #by the imitation agent, and the action took by the expert.
        loss_v = self.loss_calc(action_prob_dist, expert_actions).to(self.device)


        self.totalLoss += loss_v
        self.loss = loss_v


        loss_v.backward()
        imitation_agent.optimizer.step()


    def get_action(self, agent, observation, step_n) -> np.ndarray:
        best_act = self.find_best_action(agent.policy, observation)

        return best_act # replay buffer store lists and env does np.argmax(action)

    def find_best_action(self, network, observation: np.ndarray) -> np.ndarray:

        return self.action2one_hot(np.argmax(network(torch.tensor(observation).float()).detach()).item())


    def find_best_expert_action(self, network, observation: np.ndarray) -> np.ndarray:

        obs_v = torch.tensor(observation).float().to(self.device)
        best_q, best_act_v = float('-inf'), torch.zeros(self.action_space).to(self.device)
        for i in range(self.action_space):
            act_v = self.action2one_hot_v(i)
            q_val = network(torch.cat([obs_v, act_v.to(self.device)]))
            if q_val > best_q:
                best_q = q_val
                best_act_v = act_v
        best_act = best_act_v

        return np.argmax(best_act)

    '''def find_best_actions(self,network,observations) -> torch.tensor:

        z = torch.FloatTensor()

        for observation in observations:
            z = torch.cat(z,self.find_best_expert_action(network,observation))

        return z'''

    def action2one_hot(self, action_idx: int) -> np.ndarray:
        z = np.zeros(self.action_space)
        z[action_idx] = 1
        return z

    def action2one_hot_v(self, action_idx: int) -> torch.tensor:
        z = torch.zeros(self.action_space)
        z[action_idx] = 1
        return z

    def get_loss(self):
        return self.loss

    def get_average_loss(self, step):
        average = self.totalLoss/step
        self.totalLoss = 0
        return average
