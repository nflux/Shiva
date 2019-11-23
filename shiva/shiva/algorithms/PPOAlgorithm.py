import numpy as np
import torch
import utils.Noise as noise
from agents.PPOAgent import PPOAgent
from .Algorithm import Algorithm
from torch.distributions import Categorical


class PPOAlgorithm(Algorithm):
    def __init__(self,obs_space, acs_space, action_space_discrete,action_space_continuous,configs):

        super(PPOAlgorithm, self).__init__(obs_space,acs_space,configs)

        self.epsilon_clip = configs[0]['epsilon_clip']
        self.gamma = configs[0]['gamma']
        self.actor_loss = 0
        self.critic_loss = 0
        self.acs_space = acs_space
        self.obs_space = obs_space
        self.acs_discrete = action_space_discrete
        self.acs_continuous = action_space_continuous


    def update(self, agent, old_agent,minibatch, step_count):
        '''
            Getting a Batch from the Replay Buffer
            '''
        agent.optimizer.zero_grad()
            # Batch of Experiences
        states, old_actions, rewards, old_logprobs, next_states, dones = minibatch
        new_actions = agent.get_action(states)
        new_logprobs, entropy = torch.tensor(agent.get_logprobs(states,new_actions)).float()


        # Make everything a tensor and send to gpu if available
        states = torch.tensor(states,requires_grad=True).to(self.device)
        old_actions = torch.tensor(old_actions.astype(np.float32),requires_grad=True).to(self.device)
        rewards = torch.tensor(rewards,requires_grad=True).to(self.device)
        #We will use the next states to get the value of the next state for our advantage function
        next_states = torch.tensor(next_states,requires_grad=True).to(self.device)
        #Normalizing the rewards
        #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        old_logprobs = torch.tensor(old_logprobs,requires_grad=True).to(self.device).float()
        dones_mask = torch.ByteTensor(dones).to(self.device)


        state_values = agent.get_values(states,old_actions).requires_grad_(True)
        #Used for advantage function
        new_actions = agent.get_action(next_states)
        next_state_values = old_agent.get_values(next_states,new_actions).requires_grad_(True)


        advantage = rewards + (self.gamma*next_state_values) - state_values.detach()
        print('New logprobs: ', new_logprobs)
        print('Old logprobs: ', old_logprobs)
        ratios = torch.exp(new_logprobs - old_logprobs.detach())

        surr1 = ratios * advantage
        surr2 = torch.clamp(ratios,1.0-self.epsilon_clip,1.0+self.epsilon_clip) * advantage
        self.actor_loss = (-torch.min(surr1,surr2) -(self.configs[0]['beta']*entropy)).mean()
        self.actor_loss.backward(retain_graph=True)
        agent.optimizer.step()

        expected_state_action_values = next_state_values * self.gamma + rewards
        agent.optimizer_critic.zero_grad()

        self.critic_loss = self.loss_calc(state_values, rewards)
        self.critic_loss.backward()
        agent.optimizer_critic.step()


    def get_actor_loss(self):
        return self.actor_loss

    def get_critic_loss(self):
        return self.critic_loss

    def create_agent(self):
        self.agent = PPOAgent(self.id_generator(), self.obs_space, self.acs_discrete,self.acs_continuous, self.configs[1],self.configs[2])
        return self.agent
