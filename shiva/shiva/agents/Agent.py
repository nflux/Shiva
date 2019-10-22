import torch
import numpy as np
import helpers.misc as misc

class Agent(object):
    def __init__(self, agent_id, config):
        '''
        Base Attributes of Agent
            agent_id = given by the learner
            observation_space
            act_dim
            policy = Neural Network Policy
            target_policy = Target Neural Network Policy
            optimizer = Optimier Function
            learning_rate = Learning Rate
        '''
        {setattr(self, k, v) for k,v in config.items()}
        self.agent_id = agent_id
        self.policy = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __str__(self):
        return "<{}:id={}>".format(self.__class__, self.id)
    
    def save(self, save_path, step):
        torch.save(self.policy, save_path + '/policy.pth')

    def load_net(self, load_path):
        self.policy = torch.load(load_path)

    def find_best_action(self, network, observation) -> np.ndarray:
        '''
            Iterates over the action space to find the one with the highest Q value

            Input
                network         policy network to be used
                observation     observation from the environment
            
            Returns
                A one-hot encoded list
        '''
        obs_v = torch.tensor(observation).float().to(self.device)
        best_q, best_act_v = float('-inf'), torch.zeros(self.action_space).to(self.device)
        for i in range(self.action_space):
            act_v = misc.action2one_hot_v(self.action_space, i)
            q_val = network(torch.cat([obs_v, act_v.to(self.device)]))
            if q_val > best_q:
                best_q = q_val
                best_act_v = act_v
        best_act = best_act_v.tolist()
        return best_act