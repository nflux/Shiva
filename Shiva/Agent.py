import numpy as np
import uuid
class Agent:
    def __init__(self, obs_dim, action_dim, uid):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.id = uuid.uuid4()
        self.network = Network(self.obs_dim,self.action_dim)
        self.policy = None
        self.target_policy = None
       
    
    def policy(self, obs):
        rand = np.random.random()
        tensor = self.network.Forward(obs)
        if np.random.ramdon() < 1 - self.epsilon:

        else:
            
      
        '''
            Obtain Action from current policy. 
        '''
        return action

    def save(self):
         '''
            Save the current Obs and Action to obs_
        '''
        pass

    def load(self):
        '''
        Load the Cunrrent action load the Replay Buffer
        '''

        pass



class DQAgent:
    def __init__(self, obs_dim, action_dim, uid):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.id = uuid.uuid4()
        self.network = Network(self.obs_dim,self.action_dim)
        self.policy = None
        self.target_policy = None
       
    
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