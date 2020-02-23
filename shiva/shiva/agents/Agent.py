import torch
import torch.nn
import numpy as np
import shiva.helpers.misc as misc

class Agent(object):
    
    def __init__(self, id, obs_space, acs_space, agent_config, network_config):
        super(Agent, self).__init__()
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

    # def save(self, save_path, step):
    #     '''
    #         Do something like
    #             torch.save(self.policy, save_path + '/policy.pth')
    #             torch.save(self.critic, save_path + '/critic.pth')
    #         Or as many policies the Agent has

    #         Important:
    #             Maintain the .pth file name to have the same name as the Agent attribute
    #     '''
    #     assert False, "Method Not Implemented"

    def load_net(self, policy_name, policy_file):
        '''
            TBD
            Load as many policies the Agent needs (actor, critic, target, etc...) on the agents folder @load_path

            Do something like
                self.policy = torch.load(load_path)
            OR maybe even better:
                setattr(self, policy_name, torch.load(policy_file))

            Possible approach:
            - ShivaAdmin finds the policies saved for the Agent and calls this method that many times

        '''
        # assert False, "Method Not Implemented"
        setattr(self, policy_name, torch.load(policy_file))

    def get_action(self, obs):
        assert False, "Method Not Implemented"

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
        for i in range(self.acs_space):
            act_v = misc.action2one_hot_v(self.acs_space, i).to(self.device)
            q_val = network(torch.cat([obs_v, act_v]))
            if q_val > best_q:
                best_q = q_val
                best_act_v = act_v
        best_act = best_act_v.tolist()
        return best_act

    @staticmethod
    def copy_model_over(from_model, to_model):
        """
            Copies model parameters from @from_model to @to_model
        """
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())
    
    @staticmethod
    def mod_lr(optim, lr):
        for g in optim.param_groups:
            # print(g['lr'])
            g['lr'] = lr

    # def save(self, save_path, step_count):
    #     torch.save(self, save_path + 'agent.pth')

    # def load(self, save_path):
    #     self = torch.load(save_path + 'agent.pth')