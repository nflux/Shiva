import numpy as np
import torch
import torch.functional as F
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
        self.loss = 0
        self.acs_space = acs_space
        self.obs_space = obs_space
        self.acs_discrete = action_space_discrete
        self.acs_continuous = action_space_continuous


    def update(self, agent,minibatch, step_count):
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

        # Monte Carlo estimate of state rewards:
        new_rewards = []
        discounted_reward = 0
        for reward, done_mask in zip(reversed(rewards), reversed(done_masks)):
            if done_mask:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            new_rewards.insert(0, discounted_reward)
        new_rewards = torch.tensor(new_rewards).float()

        '''
            Training the Actor
        '''

        # Zero the gradient
        #agent.actor_optimizer.zero_grad()
        agent.optimizer.zero_grad()
        # Get the actions(probabilites) from the main actor
        current_actor_actions = agent.actor(states.float())
        dist = Categorical(current_actor_actions)
        actions = torch.tensor(np.argmax(actions,axis=1).numpy()).float()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        #Get the actions(probabilites) from the target actor
        target_actor_actions = agent.target_actor(states.float())
        dist = Categorical(current_actor_actions)
        old_log_probs = dist.log_prob(actions)
        #Find the ratio (pi_new / pi_old)
        ratios = torch.exp(log_probs - old_log_probs.detach())
        #ratios = current_actor_actions/ target_actor_actions.detach()
        # Calculate Q value for taking those actions in those states
        state_values = agent.critic(states.float())
        #Calculate next state values
        expected_state_action_values = state_values * self.gamma + new_rewards.unsqueeze(dim=-1)
        #Calculate the advantage term
        #advantage = (rewards + (self.gamma*new_rewards.unsqueeze(dim=-1)) - state_values).mean()
        advantage = new_rewards.unsqueeze(dim=-1) - state_values .detach()

        #Calculate objective functions
        surr1 = ratios * advantage
        surr2 = torch.clamp(ratios,1.0-self.epsilon_clip,1.0+self.epsilon_clip) * advantage
        #Set the policy loss
        policy_loss = -torch.min(surr1,surr2)
        #entropy = Categorical(current_actor_actions).entropy()
        entropy_loss = -(self.configs[0]['beta']*entropy).mean()
        value_loss = self.loss_calc(state_values, new_rewards.unsqueeze(dim=-1))
        self.loss = policy_loss.mean() + value_loss + entropy_loss
        self.loss.backward()
        agent.optimizer.step()


    def get_loss(self):
        return self.loss



    def create_agent(self):
        self.agent = PPOAgent(self.id_generator(), self.obs_space, self.acs_discrete,self.acs_continuous, self.configs[1],self.configs[2])
        return self.agent
