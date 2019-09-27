import numpy as np
import torch
import DQNet as net
import Environment as env
import uuid 
class Agent:
    def __init__(self, obs_dim, action_dim, uid, optimizer, learningrate):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.id = uuid.uuid4()
        self.policy = None
        self.target_policy = None
        self.optimizer = None
        self.learningrate = learningrate



         
    

    def save(self):
        '''
        Save the Current Action and Observation pair to where?
        '''
        pass  

    def load(self):
        '''
        Load the Cunrrent action load the Replay Buffer
        '''

        pass



class DQAgent(Agent):
    def __init__(self, obs_dim, action_dim, uid, optimizer, learningrate):
        super(DQAgent,self, obs_dim, action_dim, uid, optimizer, learningrate).__init__
        self.policy = net(action_dim,32,64,obs_dim)
        self.target_policy = net(action_dim,32, 64 obs_dim)
        self.optimizer = optimizer(params=self.policy.parameters(), lr=learningrate)
    '''
    def policy(self, obs, epsilon):
        self.epsilon = epsilon
        rand = np.random.random()
        
        if np.random.random() < 1 - self.epsilon:
            action = env.get_actions()
        else:
            obs_a = np.array([obs], copy=False)
            obs_v = torch.tensor(obs_a).to(device)
            q_vals_v = self.net.Forward(obs_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        
            Obtain Action from current policy by connect . 
        
        return action
        '''
    def save(self):
        '''
            Save the current Obs and Action
        '''

        torch.save(self.network,"/ShivaAgent"+str(self.id)+ ".pth")

    def load(self):
        '''
        Load the Cunrrent action
        '''
        torch.load(self.network,"/ShivaAgent"+str(self.id)+ ".pth")