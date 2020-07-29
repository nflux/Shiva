import numpy as np
import torch
from torch.nn.functional import softmax
from shiva.agents.DDPGAgent import DDPGAgent
from shiva.algorithms.Algorithm import Algorithm
from shiva.helpers.misc import one_hot_from_logits

class DDPGAlgorithm(Algorithm):
    def __init__(self, observation_space: int, action_space: dict, configs: dict):
        '''
            Inputs
                epsilon        (start, end, decay rate), example: (1, 0.02, 10**5)
                C              Number of iterations before the target network is updated
        '''
        super(DDPGAlgorithm, self).__init__(observation_space, action_space, configs)
        if hasattr(self, 'roles'):
            assert len(self.roles), "Single role is allowed for {}".format(str(self))
            self.action_space = self.action_space[self.roles[0]]
            self.observation_space = self.observation_space[self.roles[0]]

        self.actor_loss = torch.tensor(0)
        self.critic_loss = torch.tensor(0)
        self.set_action_space()
        self.critic_learning_rate = 0
        self.actor_learning_rate = 0
        self.exploration_epsilon = 0
        self.noise_scale = 0

        # print("Obs {} Acs {} Rew {} NextObs {} Dones {}".format(states, actions, rewards, next_states, dones))
        # print("Shapes Obs {} Acs {} Rew {} NextObs {} Dones {}".format(states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape))

        '''
            Updates starts here
        '''
        self.critic_learning_rate = self.agent.critic_learning_rate
        self.actor_learning_rate = self.agent.actor_learning_rate

        # # Send everything to gpu if available
        # states = states.squeeze(1).to(self.device)
        # actions = actions.squeeze(1).to(self.device)
        # rewards = rewards.squeeze(1).to(self.device)
        # next_states = next_states.squeeze(1).to(self.device)
        # dones = torch.tensor(dones, dtype=torch.bool).view(-1, 1).to(self.device)

        '''
            Training the Critic
        '''
        self.critic_learning_rate = agent.critic_learning_rate
        self.actor_learning_rate = agent.actor_learning_rate
        self.exploration_epsilon = agent.epsilon
        self.noise_scale = agent.noise_scale

        for i in range(self.updates):

            try:
                 '''For MultiAgentTensorBuffer - 1 Agent only here'''
                 states, actions, rewards, next_states, dones = buffer.sample(agent_id=agent.id, device=self.device)
                 dones = dones.bool()
            except:
                states, actions, rewards, next_states, dones = buffer.sample(device=self.device)
                dones = dones.byte()

            # Send everything to gpu if available
            states = states.squeeze(1).to(self.device)
            actions = actions.squeeze(1).to(self.device)
            rewards = rewards.squeeze(1).to(self.device)
            next_states = next_states.squeeze(1).to(self.device)
            dones = torch.tensor(dones, dtype=torch.bool).view(-1, 1).to(self.device)



            # print("Obs {} Acs {} Rew {} NextObs {} Dones {}".format(states, actions, rewards, next_states, dones_mask))
            # print("Shapes Obs {} Acs {} Rew {} NextObs {} Dones {}".format(states.shape, actions.shape, rewards.shape, next_states.shape, dones_mask.shape))

            assert self.a_space == "discrete" or self.a_space == "continuous" or self.a_space == "parameterized", \
                "acs_space config must be set to either discrete, continuous, or parameterized."

            '''  yes its supposed to be
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
                    discrete_actions = next_state_actions_target[:,:self.discrete]
                    one_hot_encoded_discrete_actions = one_hot_from_logits(discrete_actions)
                    next_state_actions_target = torch.cat([one_hot_encoded_discrete_actions, next_state_actions_target[:,self.discrete:]], dim=1)
                else:
                    discrete_actions = next_state_actions_target[:self.discrete]
                    one_hot_encoded_discrete_actions = one_hot_from_logits(discrete_actions)
                    next_state_actions_target = torch.cat([one_hot_encoded_discrete_actions, next_state_actions_target[self.discrete:]], dim=0)



        # The actions that target actor would do in the next state.
        next_state_actions_target = self.agent.target_actor(next_states.float(), gumbel=False)

        dims = len(next_state_actions_target.shape)

        if self.a_space == "discrete" or self.a_space == "parameterized":
            # Grab the discrete actions in the batch
            # generate a tensor of one hot encodings of the argmax of each discrete action tensors
            # concat the discrete and parameterized actions back together
            if dims == 3:
                discrete_actions = next_state_actions_target[:, :, :self.discrete].squeeze(dim=1)
                one_hot_encoded_discrete_actions = one_hot_from_logits(discrete_actions).unsqueeze(dim=1)
                next_state_actions_target = torch.cat(
                    [one_hot_encoded_discrete_actions, next_state_actions_target[:, :, self.discrete:]], dim=2)
            elif dims == 2:
                discrete_actions = next_state_actions_target[:, :self.discrete]
                one_hot_encoded_discrete_actions = one_hot_from_logits(discrete_actions)
                next_state_actions_target = torch.cat(
                    [one_hot_encoded_discrete_actions, next_state_actions_target[:, self.discrete:]], dim=1)
            else:
                discrete_actions = next_state_actions_target[:self.discrete]
                one_hot_encoded_discrete_actions = one_hot_from_logits(discrete_actions)
                next_state_actions_target = torch.cat(
                    [one_hot_encoded_discrete_actions, next_state_actions_target[self.discrete:]], dim=0)

        # The Q-value the target critic estimates for taking those actions in the next state.
        if dims == 3:
            Q_next_states_target = self.agent.target_critic( torch.cat([next_states.float(), next_state_actions_target.float()], 2) )
        elif dims == 2:
            Q_next_states_target = self.agent.target_critic( torch.cat([next_states.float(), next_state_actions_target.float()], 1) )
        else:
            Q_next_states_target = self.agent.target_critic( torch.cat([next_states.float(), next_state_actions_target.float()], 0) )

        # Sets the Q values of the next states to zero if they were from the last step in an episode.
        Q_next_states_target[dones] = 0.0
        # Use the Bellman equation.
        y_i = rewards.float() + self.gamma * Q_next_states_target.float()

        if self.a_space == 'discrete':
            actions = one_hot_from_logits(actions)
        else:
            actions = softmax(actions)

        # Get Q values of the batch from states and actions.
        if dims == 3:
            Q_these_states_main = self.agent.critic(torch.cat([states.float(), actions.float()], 2))
        elif dims == 2:
            Q_these_states_main = self.agent.critic(torch.cat([states.float(), actions.float()], 1))
        else:
            Q_these_states_main = self.agent.critic(torch.cat([states.float(), actions.float()], 0))

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
        self.agent.actor_optimizer.zero_grad()
        # Get the actions the main actor would take from the initial states
        if self.a_space == "discrete" or self.a_space == "parameterized":
            current_state_actor_actions = self.agent.actor(states.float(), gumbel=True)
        else:
            current_state_actor_actions = self.agent.actor(states.float())

        # Calculate Q value for taking those actions in those states'
        if dims == 3:
            actor_loss_value = self.agent.critic( torch.cat([states.float(), current_state_actor_actions.float()], 2) )
        elif dims == 2:
            actor_loss_value = self.agent.critic( torch.cat([states.float(), current_state_actor_actions.float()], 1) )
        else:
            actor_loss_value = self.agent.critic( torch.cat([states.float(), current_state_actor_actions.float()], 0) )

        # entropy_reg = (-torch.log_softmax(current_state_actor_actions, dim=2).mean() * 1e-3)/1.0 # regularize using logs probabilities
        # penalty for going beyond the bounded interval
        param_reg = torch.clamp((current_state_actor_actions**2)-torch.ones_like(current_state_actor_actions),min=0.0).mean()
        # Make the Q-value negative and add a penalty if Q > 1 or Q < -1 and entropy for richer exploration
        actor_loss = -actor_loss_value.mean() + param_reg # + entropy_reg
        # Backward Propogation!
        actor_loss.backward()
        # Update the weights in the direction of the gradient.
        self.agent.actor_optimizer.step()
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
            tgt_ct_state[k] = v*self.tau + (1 - self.tau)*tgt_ct_state[k]
        agent.target_critic.load_state_dict(tgt_ct_state)

    def create_agent(self, id):
        self.agent = DDPGAgent(id, self.observation_space, self.action_space, self.configs['Agent'], self.configs['Network'])
        return self.agent

    def create_agent_of_role(self, id, role):
        self.configs['Agent']['role'] = role
        return self.create_agent(id)

    def set_action_space(self):
        if self.action_space['continuous'] == 0:
            self.a_space = 'discrete'
        elif self.action_space['discrete'] == 0:
            assert "Continuous not tested yet"
            self.a_space = 'continuous'
        else:
            assert "Parametrized not supported yet"
        self.continuous = self.action_space['continuous']
        self.acs_space = self.action_space['acs_space']
        self.discrete = self.action_space['discrete']
        self.param = self.action_space['param']


    def get_metrics(self, episodic=False, agent_id=None):
        if not episodic:
            metrics = [
                ('Algorithm/Actor_Loss', self.actor_loss.item()),
                ('Algorithm/Critic_Loss', self.critic_loss.item()),
                ('Agent/Actor_Learning_Rate: ', self.actor_learning_rate),
                ('Agent/Critic_Learning_Rate: ', self.critic_learning_rate)
            ]
            # # not sure if I want this all of the time
            # for i, ac in enumerate(self.action_space['acs_space']):
            #     metrics.append(('Agent/Actor_Output_'+str(i), self.action[i]))
        else:
            metrics = [
                ('Algorithm/Actor_Loss', self.actor_loss.item()),
                ('Algorithm/Critic_Loss', self.critic_loss.item()),
                ('Actor Learning Rate: ', self.actor_learning_rate),
                ('Critic Learning Rate: ', self.critic_learning_rate),
                ('Agent Exploration Epsilon: ', self.exploration_epsilon),
                ('Agent Noise Scale: ', self.noise_scale)
            ]
        return metrics

    def __str__(self):
        return '<DDPGAlgorithm(num_updates={})>'.format(self.num_updates)
