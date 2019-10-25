import numpy as np
import torch
import random
from agents.ImitationAgent import ImitationAgent
import helpers.misc as misc
from .Algorithm import Algorithm
from settings import shiva

class DaggerAlgorithm(Algorithm):
    def __init__(self,obs_space, acs_space, configs):

        super(DaggerAlgorithm, self).__init__(obs_space, acs_space, configs)
        self.acs_space = acs_space
        self.obs_space = obs_space
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
       # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        return misc.action2one_hot(np.argmax(network(torch.tensor(observation).float()).detach()).item())


    def find_best_expert_action(self, network, observation: np.ndarray) -> np.ndarray:

        obs_v = torch.tensor(observation).float().to(self.device)
        best_q, best_act_v = float('-inf'), torch.zeros(self.action_space).to(self.device)
        for i in range(self.action_space):
            act_v = misc.action2one_hot_v(i)
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

    def get_loss(self):
        return self.loss

    def get_average_loss(self, step):
        average = self.totalLoss/step
        self.totalLoss = 0
        return average
