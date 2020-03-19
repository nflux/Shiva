import torch
import torch.nn
import numpy as np

class Agent(torch.nn.Module):

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
        torch.manual_seed(self.manual_seed)
        np.random.seed(self.manual_seed)


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
        flag = True
        while flag:
            try:
                model = getattr(self, policy_name)
                loaded_model = torch.load(policy_file)
                model.load_state_dict(loaded_model['state_dict'])
                setattr(self, policy_name, model)
                flag = False
            except:
                # try again
                pass

    def get_action(self, obs):
        assert False, "Method Not Implemented"

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
