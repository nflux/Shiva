import numpy as np
import torch
from torch.nn.functional import softmax
from shiva.utils import Noise as noise
from shiva.helpers.calc_helper import np_softmax
from shiva.agents.DDPGAgent import DDPGAgent
from shiva.algorithms.Algorithm import Algorithm
from shiva.helpers.misc import one_hot_from_logits

class DDPGAlgorithm(Algorithm):
    def __init__(self, observation_space: int, action_space: int, evaluate: bool, configs: dict):
        '''
            Inputs
                epsilon        (start, end, decay rate), example: (1, 0.02, 10**5)
                C              Number of iterations before the target network is updated
        '''
        super(DDPGAlgorithm, self).__init__(observation_space, action_space, configs)
        self.actor_loss = 0
        self.critic_loss = 0
        self.discrete = action_space['discrete']
        self.param = action_space['param']
        self.evaluate = evaluate
        # self.ou_noise = noise.OUNoise(self.discrete + self.param, self.exploration_noise)

    def update(self, agent, buffer, step_count, episodic=False):
        '''
            @buffer         buffer is a reference
        '''
        if episodic:
            '''
                DDPG updates at every step. This avoids doing an extra update at the end of an episode
                But it does reset the noise after an episode
            '''
            agent.ou_noise.reset()
            return

        if step_count < agent.exploration_steps:
            '''
                Don't update during exploration!
            '''
            return

        '''
            Updates starts here
        '''

        # print("updating!")

        states, actions, rewards, next_states, dones = buffer.sample()

        # Make everything a tensor and send to gpu if available
        states = torch.tensor(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.tensor(next_states).to(self.device)
        dones_mask = torch.tensor(dones, dtype=torch.bool).view(-1,1).to(self.device)
        # print(actions)
        # input()
        # print('from buffer:', states.shape, actions.shape, rewards.shape, next_states.shape, dones_mask.shape, '\n')

        assert self.a_space == "discrete" or self.a_space == "continuous" or self.a_space == "parameterized", \
            "acs_space config must be set to either discrete, continuous, or parameterized."

        '''
            Training the Critic
        '''
    
        # Zero the gradient
        agent.critic_optimizer.zero_grad()
        # The actions that target actor would do in the next state.
        next_state_actions_target = agent.target_actor(next_states.float(), gumbel=False)

        dims = len(next_state_actions_target.shape)

        if self.a_space == "discrete" or self.a_space == "parameterized":

            # Grab the discrete actions in the batch
            # generate a tensor of one hot encodings of the argmax of each discrete action tensors
            # concat the discrete and parameterized actions back together

            if dims == 3:
                discrete_actions = next_state_actions_target[:,:,:self.discrete].squeeze(dim=1)
                one_hot_encoded_discrete_actions = one_hot_from_logits(discrete_actions).unsqueeze(dim=1)
                next_state_actions_target = torch.cat([one_hot_encoded_discrete_actions, next_state_actions_target[:,:,self.discrete:]], dim=2)

            elif dims == 2:
                discrete_actions = next_state_actions_target[:,:self.discrete].squeeze(dim=0)
                one_hot_encoded_discrete_actions = one_hot_from_logits(discrete_actions)
                next_state_actions_target = torch.cat([one_hot_encoded_discrete_actions, next_state_actions_target[:,self.discrete:]], dim=1)
            else:
                discrete_actions = next_state_actions_target[:self.discrete]
                one_hot_encoded_discrete_actions = one_hot_from_logits(discrete_actions)
                next_state_actions_target = torch.cat([one_hot_encoded_discrete_actions, next_state_actions_target[self.discrete:]], dim=0)

       
        # print(next_state_actions_target.shape, '\n')

        # The Q-value the target critic estimates for taking those actions in the next state.
        if dims == 3:
            Q_next_states_target = agent.target_critic( torch.cat([next_states.float(), next_state_actions_target.float()], 2) )
        elif dims == 2:
            Q_next_states_target = agent.target_critic( torch.cat([next_states.float(), next_state_actions_target.float()], 1) )
        else:
            Q_next_states_target = agent.target_critic( torch.cat([next_states.float(), next_state_actions_target.float()], 0) )

        # Sets the Q values of the next states to zero if they were from the last step in an episode.
        Q_next_states_target[dones_mask] = 0.0
        # Use the Bellman equation.
        y_i = rewards.unsqueeze(dim=-1) + self.gamma * Q_next_states_target
        # Get Q values of the batch from states and actions.

        if self.a_space == 'discrete':
            actions = one_hot_from_logits(actions)
            # print(actions)

        # Grab the discrete actions in the batch
        if dims == 3:
            # print(states.shape, actions.shape)
            Q_these_states_main = agent.critic( torch.cat([states.float(), actions.unsqueeze(dim=1).float()], 2) )
        elif dims == 2:
            Q_these_states_main = agent.critic( torch.cat([states.float(), actions.float()], 1) )
        else:
            Q_these_states_main = agent.critic( torch.cat([states.float(), actions.float()], 0) )

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
        if self.a_space == "discrete" or self.a_space == "parameterized":
            current_state_actor_actions = agent.actor(states.float(), gumbel=True)
        else:
            current_state_actor_actions = agent.actor(states.float())

        # Calculate Q value for taking those actions in those states'
        if dims == 3:
            actor_loss_value = agent.critic( torch.cat([states.float(), current_state_actor_actions.float()], 2) )
        elif dims == 2:
            actor_loss_value = agent.critic( torch.cat([states.float(), current_state_actor_actions.float()], 1) )
        else:
            actor_loss_value = agent.critic( torch.cat([states.float(), current_state_actor_actions.float()], 0) )

        # entropy_reg = (-torch.log_softmax(current_state_actor_actions, dim=2).mean() * 1e-3)/1.0 # regularize using log probabilities
        # penalty for going beyond the bounded interval
        param_reg = torch.clamp((current_state_actor_actions**2)-torch.ones_like(current_state_actor_actions),min=0.0).mean()
        # Make the Q-value negative and add a penalty if Q > 1 or Q < -1 and entropy for richer exploration
        actor_loss = -actor_loss_value.mean() + param_reg #+ entropy_reg
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

    def create_agent(self, id=0):
        self.agent = DDPGAgent(id, self.obs_space, self.discrete + self.param, self.discrete, self.configs[1], self.configs[2])
        return self.agent

    def get_metrics(self, episodic=False):
        if not episodic:
            metrics = [
                ('Algorithm/Actor_Loss', self.actor_loss),
                ('Algorithm/Critic_Loss', self.critic_loss)
            ]
            # # not sure if I want this all of the time
            # for i, ac in enumerate(self.action_space['acs_space']):
            #     metrics.append(('Agent/Actor_Output_'+str(i), self.action[i]))
        else:
            metrics = []
        return metrics

    def __str__(self):
        return 'DDPGAlgorithm'