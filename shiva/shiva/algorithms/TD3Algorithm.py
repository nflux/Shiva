import numpy as np
import torch
import time

from shiva.agents.TD3Agent import TD3Agent
from shiva.algorithms.Algorithm import Algorithm
# from utils.Gaussian_Exploration import Gaussian_Exploration
# from utils.OU_Noise_Exploration import OU_Noise_Exploration
# from shiva.utils import Noise as noise

class TD3Algorithm(Algorithm):
    def __init__(self, observation_space: int, action_space: int, configs: dict):
        super(TD3Algorithm, self).__init__(observation_space, action_space, configs)
        torch.manual_seed(self.manual_seed)
        np.random.seed(self.manual_seed)

        self.obs_space = observation_space
        self.acs_space = action_space['acs_space']
        # self.exploration_strategy = OU_Noise_Exploration(action_space, self.configs[0])
        # self.exploration_strategy_critic = Gaussian_Exploration(self.configs[0])
        # self.ou_noise = noise.OUNoise(self.acs_space, self.noise_scale, self.noise_mu, self.noise_theta, self.noise_sigma)
        # self.ou_noise_critic = noise.OUNoise(self.acs_space, self.noise_scale, self.noise_mu, self.noise_theta, self.noise_sigma)

        self.actor_loss = 0
        self.critic_loss_1 = 0
        self.critic_loss_2 = 0

    def update(self, agent, buffer, step_count, episodic=False):
        '''
            NOTE
            If doing episodic updates only, step_count is the number of episodes actually!!!
        '''
        # if episodic:
        #     agent.ou_noise.reset()
        #     agent.ou_noise_critic.reset()
        #     return

        # if step_count < self.configs['Agent']['exploration_steps']:
        #     '''Avoid updating during exploration'''
        #     pass

        '''
            Update starts here
        '''

        self.agent = agent
        for _ in range(self.update_iterations):

            try:
                '''For MultiAgentTensorBuffer - 1 Agent only here'''
                states, actions, rewards, next_states, dones = buffer.sample(agent_id=agent.id, device=self.device)
                # dones_mask = dones.bool()
                dones_mask = dones.float()
            except:
                states, actions, rewards, next_states, dones_mask = buffer.sample(device=self.device)
                rewards = rewards.view(-1, 1)
                dones_mask = dones_mask.view(-1, 1).float()

            # states = torch.tensor(states, dtype=torch.float).to(self.device)
            # actions = torch.tensor(actions, dtype=torch.float).to(self.device)
            # rewards = torch.tensor(rewards, dtype=torch.float).view(-1,1).to(self.device)
            # next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
            # dones_mask = torch.tensor(dones, dtype=torch.float).view(-1,1).to(self.device)
            # print('from buffer:', states.shape, actions.shape, rewards.shape, next_states.shape, dones_mask.shape, '\n')
            # print('from buffer:', states, actions, rewards, next_states, dones_mask, '\n')

            # print('from buffer Actions {}', actions, '\n')
            # print('from buffer Rewards {}', rewards, '\n')
            '''
                Training the Critic
            '''
            critic_targets_next =  self.compute_critic_values_for_next_states(next_states)
            critic_targets = self.compute_critic_values_for_current_states(rewards, critic_targets_next, dones_mask)

            critic_expected_1 = self.agent.critic(torch.cat((states, actions), 1))
            critic_expected_2 = self.agent.critic_2(torch.cat((states, actions), 1))

            self.critic_loss_1 = self.loss_calc(critic_expected_1, critic_targets)
            self.critic_loss_2 = self.loss_calc(critic_expected_2, critic_targets)

            self.take_optimisation_step(self.agent.critic_optimizer, self.agent.critic, self.critic_loss_1, self.critic_grad_clip_norm)
            self.take_optimisation_step(self.agent.critic_optimizer_2, self.agent.critic_2, self.critic_loss_2, self.critic_grad_clip_norm)

            '''
                Train the Actor
            '''
            # if self.done: #we only update the learning rate at end of each episode
            #     self.update_learning_rate(self.hyperparameters["Actor"]["learning_rate"], self.actor_optimizer)
            self.actor_loss = self.calculate_actor_loss(states)
            self.take_optimisation_step(self.agent.actor_optimizer, self.agent.actor, self.actor_loss, self.actor_grad_clip_norm)

            '''
                Target updates
            '''
            # if step_count % self.c == 0:
            self.soft_update_of_target_network(self.agent.critic, self.agent.target_critic, self.critic_soft_update)
            self.soft_update_of_target_network(self.agent.critic_2, self.agent.target_critic_2, self.critic_soft_update)
            self.soft_update_of_target_network(self.agent.actor, self.agent.target_actor, self.actor_soft_update)

    def calculate_actor_loss(self, states):
        """Calculates the loss for the actor"""
        actions_pred = self.agent.actor(states)
        actor_loss = -self.agent.critic(torch.cat((states, actions_pred), 1)).mean()
        return actor_loss

    def compute_critic_values_for_next_states(self, next_states):
        """Computes the critic values for next states to be used in the loss for the critic"""
        with torch.no_grad():
            actions_next = self.agent.actor(next_states.to(self.device))
            # actions_next_with_noise =  self.exploration_strategy_critic.perturb_action_for_exploration_purposes({"action": actions_next})
            actions_next_with_noise = actions_next + torch.tensor(self.agent.ou_noise_critic.noise(), dtype=torch.float).to(self.device)
            critic_targets_next_1 = self.agent.target_critic(torch.cat((next_states, actions_next_with_noise), 1))
            critic_targets_next_2 = self.agent.target_critic_2(torch.cat((next_states, actions_next_with_noise), 1))
            critic_targets_next = torch.min(torch.cat((critic_targets_next_1, critic_targets_next_2),1), dim=-1)[0].unsqueeze(-1)
        return critic_targets_next

    def compute_critic_values_for_current_states(self, rewards, critic_targets_next, dones):
        """Computes the critic values for current states to be used in the loss for the critic"""
        critic_targets_current = rewards + (self.gamma * critic_targets_next * (1.0 - dones))
        return critic_targets_current

    def take_optimisation_step(self, optimizer, network, loss, grad_clip_norm=None, retain_graph=False):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        # if not isinstance(network, list): network = [network]
        optimizer.zero_grad() #reset gradients to 0
        loss.backward(retain_graph=retain_graph) #this calculates the gradients
        # self.logger.info("Loss -- {}".format(loss.item()))
        # if self.debug_mode: self.log_gradient_and_weight_information(network, optimizer)
        if grad_clip_norm is not None:
            # for net in network:
                # torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_norm) #clip gradients to help stabilise training
            torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip_norm) #clip gradients to help stabilise training
        optimizer.step() #this applies the gradients

    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    # Gets actions with a linearly decreasing e greedy strat
    # def get_action(self, agent, observation, step_count) -> np.ndarray: # maybe a torch.tensor
        # print('User agent.get_action!')
        # if step_count < self.exploration_steps:
        #     self.action = np.array([np.random.uniform(-1, 1) for _ in range(self.acs_space)])
        #     return self.action
        # else:
        #     """Picks an action using the actor network and then adds some noise to it to ensure exploration"""
        #     self.agent.actor.eval()
        #     with torch.no_grad():
        #         obs = torch.tensor(observation, dtype=torch.float).to(self.device)
        #         self.action = self.agent.actor(obs).cpu().data.numpy()
        #     self.agent.actor.train()
        #     # action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action": action})
        #     self.action += self.ou_noise.noise()
        #     return self.action

    def create_agent(self, id):
        self.agent = TD3Agent(id, self.obs_space, self.acs_space, self.configs['Agent'], self.configs['Network'])
        return self.agent

    def get_metrics(self, episodic=False):
        metrics = [
            ('Algorithm/Actor_Loss', self.actor_loss.item()),
            ('Algorithm/Critic_Loss', self.critic_loss_2.item()),
            ('Algorithm/Critic_2_Loss', self.critic_loss_2.item())
        ]
        for i, ac in enumerate(self.agent.action):
            metrics.append(('Agent/Actor_Output_' + str(i), ac))
        return metrics
        # if not episodic:
        #     metrics = [
        #         ('Algorithm/Actor_Loss', self.actor_loss),
        #         ('Algorithm/Critic_Loss', self.critic_loss_2),
        #         ('Algorithm/Critic_2_Loss', self.critic_loss_2)
        #     ]
        #     for i, ac in enumerate(self.agent.action):
        #         metrics.append(('Agent/Actor_Output_'+str(i), self.agent.action[i]))
        # else:
        #     pass
        # return metrics

    def __str__(self):
        return 'TD3Algorithm'