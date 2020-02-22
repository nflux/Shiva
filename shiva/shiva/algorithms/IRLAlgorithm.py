import numpy as np
import torch
import random

from shiva.algorithms.Algorithm import Algorithm
from shiva.agents.IRLAgent import IRLAgent
from shiva.helpers.misc import action2one_hot

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


class IRLAlgorithm(Algorithm):
    def __init__(self, obs_space, acs_space, configs):
        '''
            Inputs
                epsilon        (start, end, decay rate), example: (1, 0.02, 10**5)
                C              Number of iterations before the target network is updated
        '''
        super(IRLAlgorithm, self).__init__(obs_space, acs_space, configs)
        torch.manual_seed(self.manual_seed)
        np.random.seed(self.manual_seed)
        self.acs_space = acs_space['acs_space']
        self.obs_space = obs_space
        self.loss = 0
        self.expert = torch.load(self.expert_path)

    def update(self, agent, buffer, step_n, episodic=False):
        '''
            This update could be done episodically or stepwise, I don't think it matters but something to be considered
            is whether or not it will be done in batches or one data point at a time if that makes sense.

            Implementation
                1) Calculate what the current expected Q val from each sample on the replay buffer would be
                2) Calculate loss between current and past reward
                3) Optimize
                4) If agent steps reached C, update agent.target network

            Input
                agent       Agent reference who we are updating
                buffer      Buffer reference to sample from
                step_n      Current step number
                episodic    Flag indicating if update is episodic
        '''
        if episodic:
            '''
                DQN updates at every timestep, here we avoid doing an extra update after the episode terminates
            '''
            return

        states, actions, rewards, next_states, dones = buffer.sample()

        # make tensors as needed
        states_v = torch.tensor(states).float().to(self.device)
        next_states_v = torch.tensor(next_states).float().to(self.device)
        actions_v = torch.tensor(actions).to(self.device)
        rewards_v = torch.tensor(rewards).to(self.device).view(-1, 1)
        done_mask = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # print('from buffer:', states_v.shape, actions_v.shape, rewards_v.shape, next_states_v.shape, done_mask.shape, '\n')
        # input()

        agent.optimizer.zero_grad()
        # 1) GRAB Q_VALUE(s_j, a_j) from minibatch
        # input_v = torch.tensor([np.concatenate([s_i, a_i]) for s_i, a_i in zip(states, actions)]).float().to(
        #     self.device)

        state_action_values = agent.policy(input_v)
        # 2) GRAB MAX[Q_HAT_VALUES(s_j+1)]
        # For the observations s_j+1, select an action using the Policy and calculate Q values of those using the Target net

        input_v = torch.tensor([np.concatenate([s_i, agent.find_best_action(agent.target_policy, s_i)]) for s_i in
                                next_states]).float().to(self.device)

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

        # This might be useful if I decided to combine all the loss and use the target network for
        # Calculating an additional loss value. Might stabilize the reward function.
        if step_n % self.c == 0:
            agent.target_policy.load_state_dict(agent.policy.state_dict())  # Assuming is PyTorch!

    def assess_actions_by_expert(self, sample):
        '''

            This function will use the expert to identify whether actions were expert actions given the state

        '''

        # for state, action in sample:
        #     get expert action
        #     check if actions match
        #     create triple with s (ppo_action, expert_action)
        # create
        pass

    def get_loss(self):
        return self.loss

    def create_agent(self, agent_id):
        self.agent = IRLAgent(agent_id, self.obs_space, self.acs_space, self.configs[1], self.configs[2])
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