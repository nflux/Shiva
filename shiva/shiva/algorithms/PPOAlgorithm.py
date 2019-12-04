import numpy as np
import torch
import torch.functional as F
import utils.Noise as noise
from agents.PPOAgent import PPOAgent
from .Algorithm import Algorithm
from torch.distributions import Categorical
import copy


class PPOAlgorithm(Algorithm):
    def __init__(self,obs_space, acs_space, action_space_discrete,action_space_continuous,configs):

        super(PPOAlgorithm, self).__init__(obs_space,acs_space,configs)

        self.epsilon_clip = configs[0]['epsilon_clip']
        self.gamma = configs[0]['gamma']
        self.gae_lambda = configs[0]['lambda']
        self.policy_loss = 0
        self.value_loss = 0
        self.entropy_loss = 0
        self.loss = 0
        self.acs_space = acs_space
        self.obs_space = obs_space
        self.acs_discrete = action_space_discrete
        self.acs_continuous = action_space_continuous
        torch.manual_seed(configs[0]['manual_seed'])


    def update(self, agent,old_agent,minibatch, step_count):
        '''
            Getting a Batch from the Replay Buffer
        '''

        # Batch of Experiences
        states, actions, rewards, next_states, dones = minibatch

        # Make everything a tensor and send to gpu if available
        states = torch.tensor(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.tensor(next_states).to(self.device)
        done_masks = torch.ByteTensor(dones).to(self.device)

        values = agent.critic(states.float())
        actions = torch.tensor(np.argmax(actions,axis=-1).numpy()).long()

        for epoch in range(self.configs[0]['update_epochs']):

            new_rewards = []
            advantage = []
            delta= 0
            gae = 0
            for reward, val, next_val, done_mask in zip(reversed(rewards),reversed(values), reversed(values),reversed(done_masks)):
                if done_mask:
                    delta = reward - val
                    gae = delta
                else:
                    delta = reward + (self.gamma * next_val) - val
                    gae = delta + (self.gamma * self.gae_lambda * gae)
                advantage.insert(0,gae)
                new_rewards.insert(0,gae + val)
            new_rewards = torch.tensor(new_rewards).float()
            advantage = torch.tensor(advantage).float()
                    #advantages = (rewards + (self.gamma*new_rewards.unsqueeze(dim=-1)) - values)
                    #advantage = new_rewards - values
            advantage = (advantage - torch.mean(advantage)) / torch.std(advantage)
            old_action_probs = old_agent.actor(states.float())
            dist = Categorical(old_action_probs)
            old_log_probs = dist.log_prob(actions)
                    #advantage = new_rewards.unsqueeze(dim=-1) - values.detach()

            agent.optimizer.zero_grad()
            current_action_probs = agent.actor(states.float())
            dist2 = Categorical(current_action_probs)
            log_probs = dist2.log_prob(actions)
            entropy = dist2.entropy()
            #Find the ratio (pi_new / pi_old)
            ratios = torch.exp(log_probs - old_log_probs.detach())
            #advantage = (rewards + (self.gamma*new_rewards.unsqueeze(dim=-1)) - values)
            #Calculate objective functions
            surr1 = ratios * advantage
            surr2 = torch.clamp(ratios,1.0-self.epsilon_clip,1.0+self.epsilon_clip) * advantage
            #Set the policy loss
            self.policy_loss = -torch.min(surr1,surr2).mean()
            #entropy = Categorical(current_actor_actions).entropy()

            self.entropy_loss = -(self.configs[0]['beta']*entropy).mean()
            self.value_loss = self.loss_calc(values, new_rewards.unsqueeze(dim=-1))
            self.loss = self.policy_loss + self.value_loss + self.entropy_loss
            self.loss.backward(retain_graph = True)
            agent.optimizer.step()




    def get_actor_loss(self):
        return self.actor_loss

    def get_loss(self):
        return self.loss

    def get_critic_loss(self):
        return self.critic_loss

    def create_agent(self):
        self.agent = PPOAgent(self.id_generator(), self.obs_space, self.acs_discrete,self.acs_continuous, self.configs[1],self.configs[2])
        return self.agent
