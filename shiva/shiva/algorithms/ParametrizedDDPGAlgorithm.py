import numpy as np
np.random.seed(5)
import torch
torch.manual_seed(5)
import utils.Noise as noise
from helpers.calc_helper import np_softmax
from agents.ParametrizedDDPGAgent import ParametrizedDDPGAgent
from .Algorithm import Algorithm
from settings import shiva

class ParametrizedDDPGAlgorithm(Algorithm):
    def __init__(self, observation_space: int, action_space: int, configs: dict):
        '''
            Inputs
                epsilon        (start, end, decay rate), example: (1, 0.02, 10**5)
                C              Number of iterations before the target network is updated
        '''
        super(ParametrizedDDPGAlgorithm, self).__init__(observation_space, action_space, configs)
        self.scale = 0.9
        self.ou_noise = noise.OUNoise(action_space['discrete']+action_space['param'], self.scale)
        self.actor_loss = 0
        self.critic_loss = 0


    def update(self, agent, minibatch, step_count):

        '''
            Getting a Batch from the Replay Buffer
        '''
        # print('update')
        # Batch of Experiences
        states, actions, rewards, next_states, dones = minibatch

        # Make everything a tensor and send to gpu if available
        states = torch.tensor(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.tensor(next_states).to(self.device)
        dones_mask = torch.tensor(dones, dtype=np.bool).view(-1,1).to(self.device)
        print('from buffer:', states.shape, actions.shape, rewards.shape, next_states.shape, dones_mask.shape, '\n')
        '''
            Training the Critic
        '''

        # Zero the gradient
        agent.critic_optimizer.zero_grad()
        # The actions that target actor would do in the next state.
        next_state_actions_target = agent.target_actor(next_states.float(), hot=True)
        # print(next_state_actions_target.shape, '\n')
        # The Q-value the target critic estimates for taking those actions in the next state.
        Q_next_states_target = agent.target_critic( torch.cat([next_states.float(), next_state_actions_target.float()], 2) )
        # Sets the Q values of the next states to zero if they were from the last step in an episode.
        Q_next_states_target[dones_mask] = 0.0
        # Use the Bellman equation.
        y_i = rewards.unsqueeze(dim=-1) + self.gamma * Q_next_states_target
        # Get Q values of the batch from states and actions.
        Q_these_states_main = agent.critic( torch.cat([states.float(), actions.float()], 2) )
        # Calculate the loss.
        critic_loss = self.loss_calc(y_i.detach(), Q_these_states_main)
        # Backward propogation!
        critic_loss.backward()
        # Update the weights in the direction of the gradient.
        agent.critic_optimizer.step()
        # Save critic loss for tensorboard
        self.critic_loss = critic_loss

        '''
            Training the Actor
        '''

        # Zero the gradient
        agent.actor_optimizer.zero_grad()
        # Get the actions the main actor would take from the initial states
        current_state_actor_actions = agent.actor(states.float(), hot=True)
        # Calculate Q value for taking those actions in those states
        actor_loss_value = agent.critic( torch.cat([states.float(), current_state_actor_actions.float()], 2) )
        # might not be perfect, needs to be tested more
        entropy_reg = (-torch.log_softmax(current_state_actor_actions, dim=2).mean() * 1e-3)/1.0 # regularize using log probabilities
        # print(entropy_reg)
        # penalty for going beyond the bounded interval
        param_reg = torch.clamp((current_state_actor_actions**2)-torch.ones_like(current_state_actor_actions),min=0.0).mean()
        # Make the Q-value negative and add a penalty if Q > 1 or Q < -1 and entropy for richer exploration
        actor_loss = -actor_loss_value.mean() + param_reg + entropy_reg
        # Backward Propogation!
        actor_loss.backward()
        # Update the weights in the direction of the gradient.
        agent.actor_optimizer.step()
        # Save actor loss for tensorboard
        self.actor_loss = actor_loss

        '''
            Soft Target Network Updates
        '''

        # Update Target Actor
        ac_state = agent.actor.state_dict()
        tgt_ac_state = agent.target_actor.state_dict()

        for k, v in ac_state.items():
            tgt_ac_state[k] = v*self.tau + (1 - self.tau)*tgt_ac_state[k] 
        agent.target_actor.load_state_dict(tgt_ac_state)

        # Update Target Critic
        ct_state = agent.critic.state_dict()
        tgt_ct_state = agent.target_critic.state_dict()

        for k, v in ct_state.items():
            tgt_ct_state[k] =  v*self.tau + (1 - self.tau)*tgt_ct_state[k] 
        agent.target_critic.load_state_dict(tgt_ct_state)

        '''
            Hard Target Network Updates
        '''

        # if step_count % 1000 == 0:

        #     for target_param,param in zip(agent.target_critic.parameters(), agent.critic.parameters()):
        #         target_param.data.copy_(param.data)

        #     for target_param,param in zip(agent.target_actor.parameters(), agent.actor.parameters()):
        #         target_param.data.copy_(param.data)

        return agent

    # Gets actions with a linearly decreasing e greedy strat
    def get_action(self, agent, observation, step_count) -> np.ndarray: # maybe a torch.tensor
        # print('get action')
        if step_count < self.exploration_steps:

            action = np.array([np.random.uniform(0,1) for _ in range(self.acs_space['discrete']+self.acs_space['param'])])
            action += self.ou_noise.noise()
            action = np.concatenate([ np_softmax(action[:self.acs_space['discrete']]), action[self.acs_space['discrete']:] ])
            action = np.clip(action, -1, 1)
            # print('random action shape', action[:self.acs_space['discrete']].sum(), action.shape)
            return action

        else:

            self.ou_noise.set_scale(0.1)
            observation = torch.tensor([observation]).to(self.device)
            action = agent.get_action(observation.float()).cpu().data.numpy()

            # useful for debugging
            if step_count % 100 == 0:
                # print(action)
                pass
            # action += self.ou_noise.noise()
            action = np.clip(action, -1,1)
            print('actor action shape', action.shape)
        
            return action[0, 0] # timestamp 0, agent 0

    def create_agent(self, id):
        # print(self.obs_space)
        # input()
        self.agent = ParametrizedDDPGAgent(id, self.obs_space, self.acs_space['discrete']+self.acs_space['param'], self.acs_space['discrete'], self.configs[1], self.configs[2])
        return self.agent

    def get_actor_loss(self):
        return self.actor_loss

    def get_critic_loss(self):
        return self.critic_loss