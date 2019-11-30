import torch
import numpy as np
import helpers.misc as misc

class Agent(object):
    def __init__(self, id, obs_space, acs_space, agent_config, network_config):
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
        {setattr(self, k, v) for k,v in agent_config.items()}
        self.obs_space = obs_space
        self.acs_space = acs_space
        self.optimizer_function = getattr(torch.optim, agent_config['optimizer_function'])
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
        best_q, best_act_v = float('-inf'), torch.zeros(self.acs_space).to(self.device)
        # print(self.acs_space)
        for i in range(self.acs_space):
            act_v = misc.action2one_hot_v(self.acs_space, i).to(self.device) 
            # print(obs_v.shape, act_v.shape)
            # print(network)
            q_val = network(torch.cat([obs_v, act_v]))
            if q_val > best_q:
                best_q = q_val
                best_act_v = act_v
        best_act = best_act_v.tolist()
        return best_act