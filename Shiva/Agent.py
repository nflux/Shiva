import numpy as np
import torch
import uuid
class Agent:
    def __init__(self, obs_dim, action_dim, uid):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.id = uuid.uuid4()
        self.network = Network(self.obs_dim,self.action_dim)
        self.policy = None
        self.target_policy = None
        self.epsilon = 0.0
       
    
    def policy(self, obs):
        rand = np.random.random()
        
        if np.random.random() < 1 - self.epsilon:
            action = Environmnet.get_action()
        else:
            obs_a = np.array([obs], copy=False)
            obs_v = torch.tensor(obs_a).to(device)
            q_vals_v = self.network.Forward(obs_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        return action
        '''
            Obtain Action from current policy. 
        '''
      

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
    def __init__(self, obs_dim, action_dim, uid):
        super(DQAgent,self, obs_dim, action_dim, uid).__init__
       
    
    def policy(self, obs):
        action = self.network.Forward(obs)
        '''
            Obtain Action from current policy. 
        '''
        return action

    def save(self):
        '''
            Save the current Obs and Action
        '''
        pass

    def load(self):
        '''
        Load the Cunrrent action
        '''

        pass