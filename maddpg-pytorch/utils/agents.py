from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
import torch
from .networks import MLPNetwork
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise
import numpy as np
class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 a_lr=0.001, c_lr=0.001, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.updating_actor = False

        self.policy = MLPNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 discrete_action=discrete_action, is_actor= True,norm_in= False,agent=self)
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,is_actor=False,norm_in= False,agent=self)
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,is_actor=True,
                                        discrete_action=discrete_action,norm_in= False,agent=self)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim,is_actor=False,norm_in= False,agent=self)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.param_dim = 5
        self.action_dim = 3
        self.critic_grad_by_action = np.zeros(self.param_dim)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=a_lr, weight_decay =0)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=c_lr, weight_decay=0)
        if not discrete_action: # input to OUNoise is size of param space # TODO change OUNoise param to # params
            self.exploration = OUNoise(self.param_dim) # hard coded
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        
        action = self.policy(obs)
        #print(action)
        # mixed disc/cont
        if explore:     
            a = action[0,:self.action_dim].view(1,self.action_dim)
            p = (action[0,self.action_dim:].view(1,self.param_dim) + Variable(Tensor(self.exploration.noise()),requires_grad=False)) # get noisey params (OU)
            action = torch.cat((a,p),1) 
        #print(action)
        return action
            
        '''if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action
'''
    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    #Used to get just the actor weights and params.
    def get_actor_params(self):
        return {'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict()}

    #Used to get just the critic weights and params.
    def get_critic_params(self):
        return {'critic': self.critic.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

