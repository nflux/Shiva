import numpy as np
import torch
import torch.functional as F
from torch.distributions import Categorical
from torch.distributions.normal import Normal
from shiva.helpers.utils import Noise as noise
from shiva.agents.PPOAgent import PPOAgent
from shiva.algorithms.Algorithm import Algorithm

class ContinuousPPOAlgorithm(Algorithm):
    def __init__(self,obs_space, acs_space, configs):
        super(ContinuousPPOAlgorithm, self).__init__(obs_space,acs_space,configs)
        torch.manual_seed(self.manual_seed)
        np.random.seed(self.manual_seed)
        #self.epsilon_clip = configs[0]['epsilon_clip']
        #self.gamma = configs[0]['gamma']
        #self.gae_lambda = configs[0]['lambda']
        #self.grad_clip = configs[0]['grad_clip']
        self.policy_loss = 0
        self.value_loss = 0
        self.entropy_loss = 0
        self.ratio_mean = 0
        self.sigma_mean = 0
        self.logstd_mean = 0
        self.mu_mean = 0
        self.loss = 0
        self.acs_space = acs_space
        self.obs_space = obs_space


    def update(self, agent,buffer, step_count,episodic=True):
        '''
            Getting a Batch from the Replay Buffer
        '''
        self.step_count = step_count
        minibatch = buffer.full_buffer()
        # Batch of Experiences
        states, actions, rewards, next_states, dones, old_log_probs = minibatch
        # Make everything a tensor and send to gpu if available
        states = torch.cat([states[i] for i in range(len(states))]).to(self.device)
        actions = torch.cat([actions[i] for i in range(len(actions))]).float().to(self.device)
        rewards = torch.cat([rewards[i] for i in range(len(rewards))]).to(self.device)
        next_states = torch.cat([next_states[i] for i in range(len(next_states))]).to(self.device)
        old_log_probs = torch.cat([old_log_probs[i] for i in range(len(old_log_probs))]).to(self.device)
        # done_masks = torch.tensor(dones, dtype=np.bool).to(self.device)
        done_masks = torch.cat([dones[i] for i in range(len(dones))]).to(self.device)

        #Calculate approximated state values and next state values using the critic
        values = agent.critic(states.float()).to(self.device)
        next_values = agent.critic(next_states.float()).to(self.device)


        #Calculate Discounted Rewards and Advantages using the General Advantage Equation
        new_rewards = []
        advantage = []
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
            advantage.insert(0,gae)


        #Format discounted rewards and advantages for torch use
        new_rewards = torch.tensor(new_rewards).float().to(self.device)
        advantage = torch.tensor(advantage).float().to(self.device)
        #Normalize the advantages
        advantage = (advantage - torch.mean(advantage)) / (torch.std(advantage) + 1e-5)
        #Temporarily used for different return types from single and multienv implementations
        '''if type(logprobs) == np.ndarray:
            old_log_probs = torch.from_numpy(logprobs).float().sum(-1,keepdim=True).detach().to(self.device)
        else:
            old_log_probs = logprobs.clone().detach().sum(-1,keepdim=True).to(self.device)'''



        #Update model weights for a configurable amount of epochs
        for epoch in range(self.update_epochs):
            indices = np.random.permutation(range(len(states)))
            for idx in np.arange(0,len(states),self.batch_size):
                values = agent.critic(states[indices[idx:idx+self.batch_size]].float())
                self.value_loss = self.value_coef* self.loss_calc(values,new_rewards[indices[idx:idx+self.batch_size]].unsqueeze(dim=-1))
                #Estmate means for approximated Normal Distributions
                mu = agent.mu(states[indices[idx:idx+self.batch_size]].float()).squeeze(0)
                #Log standard Deviations for estimated Normal Distributions
                logstd = agent.logstd.expand_as(mu)
                self.logstd_mean = logstd.mean()
                self.mu_mean = mu.mean()
                #Formatting for Distribution
                if len(mu.shape) == 2: mu = mu.squeeze(-1)
                if len(logstd.shape) == 2: logstd = logstd.squeeze(-1)
                dist= Normal(mu,logstd.exp())
                new_log_probs = dist.log_prob(actions[indices[idx:idx+self.batch_size]]).sum(-1,keepdim=True)
                entropy = dist.entropy().sum(-1).mean()
                #Ratios for PPO Objective Function
                self.ratios = torch.exp(new_log_probs.double() - old_log_probs[indices[idx:idx+self.batch_size]].double()).float()
                #Positive advantage pushes distribution towards action, negative advantage pushes away from action
                surr1 = self.ratios * advantage[indices[idx:idx+self.batch_size]].unsqueeze(dim=-1)
                surr2 = torch.clamp(self.ratios,1.0-self.epsilon_clip,1.0+self.epsilon_clip) * advantage[indices[idx:idx+self.batch_size]].unsqueeze(dim=-1)
                self.ratio_mean = self.ratios.mean()
                #Optimize Parameters
                agent.optimizer.zero_grad()
                self.entropy_loss = -(self.beta*entropy)
                self.policy_loss =  -torch.min(surr1,surr2).mean()
                self.loss =  self.policy_loss +  self.entropy_loss + self.value_loss
                self.loss.backward()
                #torch.nn.utils.clip_grad_norm_(agent.parameters(), self.grad_clip)
                agent.optimizer.step()


        print('Done updating')
        #print(len(buffer))
        buffer.clear_buffer()

    def get_metrics(self, episodic=False):
        if not episodic:
            metrics = [
                #('Algorithm/Loss_per_Step', self.loss),
                ('Algorithm/Policy_Loss_per_Step', self.policy_loss),
                ('Algorithm/Value_Loss_per_Step', self.value_loss),
                ('Algorithm/Entropy_Loss_per_Step', self.entropy_loss),
                ('Algorithm/Ratios', self.ratio_mean),
                ('Algorithm/Var', self.logstd_mean),
                ('Algorithm/Mu', self.mu_mean),
            ]
        else:
            metrics = []
        return metrics

    def get_actor_loss(self):
        return self.actor_loss

    def get_loss(self):
        return self.loss

    def get_critic_loss(self):
        return self.critic_loss

    def create_agent(self,id=0):
        self.agent = PPOAgent(id, self.obs_space, self.acs_space, self.configs['Agent'], self.configs['Network'])
        return self.agent

    def __str__(self):
        return 'ContinuousPPOAlgorithm'
