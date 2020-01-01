import numpy as np
import torch
import random

from shiva.agents.ImitationAgent import ImitationAgent
from shiva.agents.ParametrizedDDPGAgent import ParametrizedDDPGAgent
from shiva.helpers import misc
from shiva.algorithms.Algorithm import Algorithm

class ImitationAlgorithm(Algorithm):
    def __init__(self,obs_space,acs_space,configs):
        super(ImitationAlgorithm, self).__init__(obs_space, acs_space, configs)
        torch.manual_seed(self.manual_seed)
        np.random.seed(self.manual_seed)
        self.action_policy = configs[1]['action_policy']
        self.loss = 0
    
    def supervised_update(self, agent, minibatch, step_n):
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

        #actions = torch.LongTensor(np.argmax(actions,axis = 1))
        actions = torch.LongTensor(actions)
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

        if self.configs[0]['loss_function'] == 'MSELoss':
            action_prob_dist = action_prob_dist.view(actions.shape)
            actions = actions.float()
        
        self.loss = self.loss_calc(action_prob_dist, actions).to(self.device)
        self.loss.backward()
        agent.optimizer.step()


    '''def get_action(self, agent, observation) -> np.ndarray:

        best_act = self.find_best_action(agent.policy, observation)

        return best_act # replay buffer store lists and env does np.argmax(action)

    def find_best_action(self, network, observation: np.ndarray) -> np.ndarray:

        obs_v = torch.tensor(observation).float().to(self.device)
        best_q, best_act_v = float('-inf'), torch.zeros(self.acs_space).to(self.device)
        for i in range(self.acs_space):
            act_v = misc.action2one_hot_v(self.acs_space,i)
            q_val = network(torch.cat([obs_v, act_v.to(self.device)]))
            if q_val > best_q:
                best_q = q_val
                best_act_v = act_v
        best_act = best_act_v.tolist()
        return best_act'''

    def dagger_update(self, imitation_agent,expert_agent, minibatch, step_n):
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
       # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        states, actions, rewards, next_states, dones = minibatch


        rewards_v = torch.tensor(rewards).to(self.device)
        done_mask = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # zero optimizer
        imitation_agent.optimizer.zero_grad()

        input_v = torch.tensor(states).float().to(self.device)

        action_prob_dist = imitation_agent.policy(input_v)

        #initialize tensor to store expert actions queried.
        if self.configs[0]['loss_function'] == 'MSELoss':
            expert_actions = torch.zeros([len(states),self.acs_space]).float()
        elif self.configs[0]['loss_function'] == 'CrossEntropyLoss':
            expert_actions = torch.zeros([len(states)]).long()
        #place the experts actions into a single tensor.
        for i in range(len(states)):
            expert_actions[i] = torch.from_numpy(expert_agent.find_best_imitation_action(states[i]))


        #detach actions to ensure they do not interfere with the network updates.
        #CrossEntropyLoss requires target to be a long tensor
        #expert_actions = expert_actions.detach().long()

        #format the shape properly to ensure proper loss calculations
        if self.configs[0]['loss_function'] == 'MSELoss':
            if (len(actions.shape) > 1):
                action_prob_dist = action_prob_dist.view(actions.shape[0],actions.shape[len(actions.shape)-1])
            else:
                action_prob_dist = action_prob_dist.view(actions.shape[0])
            #MSELoss requires target to be a float tensor
            #expert_actions = expert_actions.float()

        #calculate loss based on loss functions dictated in the configs
        self.loss = self.loss_calc(action_prob_dist, expert_actions).to(self.device)
        self.loss.backward()
        imitation_agent.optimizer.step()


    def get_action(self, agent, observation, step_n) -> np.ndarray:
        best_act = self.find_best_action(agent.policy, observation)

        return best_act # replay buffer store lists and env does np.argmax(action)

    def find_best_action(self, network, observation: np.ndarray) -> np.ndarray:
        if self.action_policy =='argmax':
            return np.random.choice(network(torch.tensor(observation).float()).detach().numpy())
        else:
            return misc.action2one_hot(self.acs_space,np.argmax(network(torch.tensor(observation).float()).detach()).item())

    def find_best_expert_action(self, network, observation: np.ndarray) -> np.ndarray:

        obs_v = torch.tensor(observation).float().to(self.device)
        best_q, best_act_v = float('-inf'), torch.zeros(self.acs_space).to(self.device)
        for i in range(self.acs_space):
            act_v = misc.action2one_hot_v(self.acs_space,i)
            q_val = network(torch.cat([obs_v, act_v.to(self.device)]))
            if q_val > best_q:
                best_q = q_val
                best_act_v = act_v
        best_act = best_act_v

        return np.argmax(best_act)

    def get_loss(self):
        return self.loss
    
    def create_agent(self):
        new_agent = ImitationAgent(self.id_generator(),self.obs_space,self.acs_space,self.configs[1],self.configs[2])
        self.agents.append(new_agent)
        return new_agent

class ImitationRoboCupAlgorithm(Algorithm):
    def __init__(self,obs_space,acs_space,configs):
        super(ImitationRoboCupAlgorithm, self).__init__(obs_space, acs_space, configs)
        self.acs_dim = self.acs_space['discrete'] + self.acs_space['param']
        self.action_policy = configs[1]['action_policy']
        self.loss = 0
    
    def supervised_update(self, agent, minibatch, step_n):

        states, actions, rewards, next_states, dones = minibatch

        # zero optimizer
        agent.actor_optimizer.zero_grad()

        action_prob_dist = agent.actor(states)

        actions = actions.detach()

        action_prob_dist = action_prob_dist.view(actions.shape)

        self.loss = self.loss_calc(action_prob_dist, actions).to(self.device)
        # print('super_loss:', self.loss)
        self.loss.backward()
        agent.actor_optimizer.step()
    
    def dagger_update(self, imitation_agent, minibatch, step_n):

        states, actions, rewards, next_states, dones, expert_actions = minibatch

        # zero optimizer
        imitation_agent.actor_optimizer.zero_grad()

        action_prob_dist = imitation_agent.actor(states)
        # print('before', action_prob_dist)

        if (len(actions.shape) > 1):
            action_prob_dist = action_prob_dist.view(actions.shape[0],actions.shape[len(actions.shape)-1])
        else:
            action_prob_dist = action_prob_dist.view(actions.shape[0])
        
        # print('after', action_prob_dist)
        # print('exper', expert_actions)

        #calculate loss based on loss functions dictated in the configs
        self.loss = self.loss_calc(action_prob_dist, expert_actions).to(self.device)
        # print('Dagger_loss:', self.loss)
        self.loss.backward()
        imitation_agent.actor_optimizer.step()

    def create_agent(self):
        new_agent = ParametrizedDDPGAgent(self.id_generator(),self.obs_space,self.acs_space['discrete']+self.acs_space['param'],self.acs_space['discrete'],self.configs[1],self.configs[2])
        return new_agent
    
    def get_loss(self):
        return self.loss
