import numpy as np
import torch
import utils.Noise as noise
from agents.DDPGAgent import DDPGAgent
from .Algorithm import Algorithm
from random import randint

class DiscreteDDPGAlgorithm(Algorithm):
    def __init__(self, observation_space: int, action_space: int, configs: dict):
        '''
            Inputs
                epsilon        (start, end, decay rate), example: (1, 0.02, 10**5)
                C              Number of iterations before the target network is updated
        '''
        super(DiscreteDDPGAlgorithm, self).__init__(observation_space, action_space, configs)

        self.scale = 0.9
        self.ou_noise = noise.OUNoise(action_space, self.scale)
        self.actor_loss = 0
        self.critic_loss = 0


    def update(self, agent, minibatch, step_count):

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
        dones_mask = torch.tensor(dones, dtype=np.bool).view(-1,1).to(self.device)

        '''
            Training the Critic
        '''

        # Zero the gradient
        agent.critic_optimizer.zero_grad()
        # The actions that target actor would do in the next state.
        next_state_actions_target = agent.target_actor(next_states.float(), gumbel=False)
        # The Q-value the target critic estimates for taking those actions in the next state.
        Q_next_states_target = agent.target_critic(next_states.float(), next_state_actions_target.float())
        # Sets the Q values of the next states to zero if they were from the last step in an episode.
        Q_next_states_target[dones_mask] = 0.0
        # Use the Bellman equation.
        y_i = rewards.unsqueeze(dim=-1) + self.gamma * Q_next_states_target
        # Get Q values of the batch from states and actions.
        # print(actions.shape)
        Q_these_states_main = agent.critic(states.float(), actions.float().squeeze())
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
        current_state_actor_actions = agent.actor(states.float(), gumbel=True)
        # Calculate Q value for taking those actions in those states
        actor_loss_value = agent.critic(states.float(), current_state_actor_actions.float())
        # miracle line of code
        param_reg = torch.clamp((current_state_actor_actions**2)-torch.ones_like(current_state_actor_actions),min=0.0).mean()
        # Make the Q-value negative and add a penalty if Q > 1 or Q < -1
        actor_loss = -actor_loss_value.mean() + param_reg
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
            tgt_ac_state[k] = tgt_ac_state[k] * self.tau + (1 - self.tau) * v
        agent.target_actor.load_state_dict(tgt_ac_state)

        # Update Target Critic
        ct_state = agent.critic.state_dict()
        tgt_ct_state = agent.target_critic.state_dict()

        for k, v in ct_state.items():
            tgt_ct_state[k] = tgt_ct_state[k] * self.tau + (1 - self.tau) * v
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

        if step_count < self.exploration_steps:

            index = randint(0,2)
            action = [0]*3
            action[index] = 1
            action = np.array(action)
            # action += self.ou_noise.noise()
            # action = np.clip(action, -1, 1)
            return action

        else:

            self.ou_noise.set_scale(0.1)
            observation = torch.tensor(observation).to(self.device)
            action = agent.actor(observation.float()).cpu().data.numpy()
            # useful for debugging
            # maybe should change the print to a log
            if step_count % 100 == 0:
                # print(action)
                pass
            # action += self.ou_noise.noise()
            action = np.clip(action, -1,1)

            return action

    def create_agent(self, id): 
        new_agent = DDPGAgent(id, self.obs_space, self.acs_space, self.configs[1], self.configs[2])
        self.agent = new_agent
        return new_agent

    def get_actor_loss(self):
        return self.actor_loss

    def get_critic_loss(self):
        return self.critic_loss