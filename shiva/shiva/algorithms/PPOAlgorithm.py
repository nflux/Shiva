import numpy as np
import torch
import torch.functional as F
from torch.distributions import Categorical

from shiva.utils import Noise as noise
from shiva.agents.PPOAgent import PPOAgent
from shiva.algorithms.Algorithm import Algorithm

class PPOAlgorithm(Algorithm):
    def __init__(self,obs_space, acs_space, action_space_discrete,action_space_continuous,configs):

        super(PPOAlgorithm, self).__init__(obs_space,acs_space,configs)
        torch.manual_seed(self.manual_seed)
        np.random.seed(self.manual_seed)
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

        #Calculate approximated state values and next state values using the critic
        values = agent.critic(states.float())
        next_values = agent.critic(next_states.float()).to(self.device)

        actions = torch.tensor(np.argmax(actions).numpy()).float()
        target_actor_actions = old_agent.actor(states.float())
        dist2 = Categorical(target_actor_actions)
        old_log_probs = dist2.log_prob(actions)

        for epoch in range(self.configs[0]['update_epochs']):
            #Calculate Discounted Rewards and Advantages using the General Advantage Equation
            new_rewards = []
            advantages = []
            delta= 0
            gae = 0
            for i in reversed(range(len(rewards))):
                if done_masks[i]:
                    delta = rewards[i]-values[i]
                    gae = delta
                else:
                    delta = rewards[i] + self.gamma * next_values[i]  - values[i]
                    gae = delta + self.gamma * self.gae_lambda * gae
                new_rewards.insert(0,gae+values[i])
                advantages.insert(0,gae)
            #Format discounted rewards and advantages for torch use
            new_rewards = torch.tensor(new_rewards).float().to(self.device)
            advantages = torch.tensor(advantages).float().to(self.device)
            advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)

            agent.optimizer.zero_grad()
            current_actor_actions = agent.actor(states.float())
            dist = Categorical(current_actor_actions)
            #actions = torch.tensor(np.argmax(actions).numpy()).float()
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            #Get the actions(probabilites) from the target actor
            #target_actor_actions = old_agent.actor(states.float())
            #dist2 = Categorical(target_actor_actions)
            #old_log_probs = dist2.log_prob(actions)
            #Find the ratio (pi_new / pi_old)
            ratios = torch.exp(log_probs - old_log_probs.detach())
            #Calculate objective functions
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios,1.0-self.epsilon_clip,1.0+self.epsilon_clip) * advantages
            #Set the policy loss
            self.policy_loss = -torch.min(surr1,surr2).mean()
            self.entropy_loss = -(self.configs[0]['beta']*entropy).mean()
            self.value_loss = self.loss_calc(values, new_rewards.unsqueeze(dim=-1))
            self.loss = self.policy_loss + self.value_loss + self.entropy_loss
            self.loss.backward(retain_graph = True)
            agent.optimizer.step()

    def get_metrics(self, episodic=False):
        if not episodic:
            metrics = [
                ('Algorithm/Loss_per_Step', self.loss),
                ('Algorithm/Policy_Loss_per_Step', self.policy_loss),
                ('Algorithm/Value_Loss_per_Step', self.value_loss),
                ('Algorithm/Entropy_Loss_per_Step', self.entropy_loss),
            ]
        else:
            metrics = []
        return metrics


    def get_loss(self):
        return self.loss



    def create_agent(self):
        self.agent = PPOAgent(self.id_generator(), self.obs_space, self.acs_discrete,self.acs_continuous, self.configs[1],self.configs[2])
        return self.agent

    def __str__(self):
        return 'PPOAlgorithm'