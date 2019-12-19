import torch
import random
from scipy import stats
import copy
import numpy as np

from shiva.core.admin import Admin
from shiva.metalearners.MetaLearner import MetaLearner
from shiva.helpers.config_handler import load_class

class MultipleAgentMetaLearner(MetaLearner):
    def __init__(self, configs):
        super(MultipleAgentMetaLearner,self).__init__(configs)
        self.configs = configs
        self.learnerCount = 0
        self.learner_list_size = configs["MetaLearner"]["learner_list"]
        self.learners = [None] * configs['MetaLearner']['learner_list']
        self.learning_rate_range = configs["MetaLearner"]["learning_rate"]
        self.episode_rewards = list()
        self.process_list = list()
        #self.multiprocessing_learners()
        self.run()

    def run(self):
        eval_env = load_class('shiva.eval_envs', self.configs['Evaluation']["type"])
        if self.start_mode == self.EVAL_MODE:
            # self.eval_envs = []
            # Load Learners to be passed to the Evaluation
            self.learners = [ Admin._load_learner(load_path) for load_path in self.configs['Evaluation']['load_path'] ]

            #assigning a new attribute in Evaluation called learners.
            # self.configs['Evaluation']['learners'] = self.learners

            # # Create Evaluation class
            # print(self.configs['Evaluation'])
            # eval_env = getattr(eval_envs, self.configs['Evaluation']["type"])
            # print(eval_envs.GymDiscreteEvaluation(self.configs['Evaluation']))
            # self.eval_envs.append(eval_env(self.configs['Evaluation']))

            # self.eval_envs[0].evaluate_agents()
            pass


        elif self.start_mode == self.PROD_MODE:

            '''# agents, environments, algorithm, data, configs for a single agent learner
            #agents, environments, algorithm, data, config
            self.learner = self.create_learner()

            # self.learner = self.create_learner(self.agent, self.eval_env, self.algorithm, self.buffer, self.learner_config)
            shiva.add_learner_profile(self.learner)

            # initialize the learner instances
            self.learner.launch()
            shiva.update_agents_profile(self.learner)

            # Runs the learner for a number of episodes given by the config
            self.learner.run()'''

            # self.populate_learners()
            # self.eval = eval_env(self.configs['Evaluation'], self.learners)
            # self.redeployment()
            # # self.sample()

            # self.multiprocessing_learners()
            self.populate_learners()
            self.eval = eval_env(self.configs['Evaluation'], self.learners)
            self.redeployment()
                # self.sample()

            self.multiprocessing_learners()
        
        for i in range(len(self.eval.eval_scores)):
            print('Average Reward for agent: ', i, '\n', np.average(self.eval.eval_scores[i]))
        # self.save()
        print('bye')

    def redeployment (self):
        for i in self.learners:
            i.eval = self.eval
        # while True:
        #     self.learners, self.episode_rewards =  self.eval.rank_agents()
        #     self.exploitation(self.learners, self.episode_rewards)
        #     self.exploration(self.learners)
        #     self.multiprocessing_learners()



    def create_learner(self):
        self.configs['Learner']['eval'] = list()
        self.configs['Evaluation']['sample'] = False
        self.configs['Learner']['updates_per_iteration'] = self.updates_per_iteration
        learner_class = load_class('shiva.learners', self.configs['Learner']['type'])
        return learner_class(self.get_id(), self.configs)

    '''#threading learners
    def multiprocessing_learners(sLearnerelf):
        for rank in range(self.learnerList):
            p = torch.multiprocessing.Process(target=self.run)
            p.start()
            self.process_list.append(p)
        for p in self.process_list:
            p.join()'''

    def check_condition (self):
        k = True
        for i in self.learners:
            if item.ep_count == item.episodes:
                k = False
            elif item.ep_count >= item.episodes:
                k = True
            else:
                k = False
        return k

    #Run each Learner in a separate process and join the processes when they are all finished running
    def multiprocessing_learners(self):
        event = torch.multiprocessing.Event()
        self.eval.event = event

        while not all(item.ep_count >= item.episodes for item in self.learners):
            self.process_list = []
            for learner in self.learners:
                print ('id:', learner.id, 'ep_count:', learner.ep_count,'ep:', learner.episodes, 'learning rate:', learner.agent.learning_rate)
            # for learner in range(len(self.learners)):
                if learner.ep_count < learner.episodes:
                    s = "learner" + "_" + str(learner)
                    p = torch.multiprocessing.Process(name=s, target=self.run_learner(learner.id))
                    p.start()
                    self.process_list.append(p)
            
            print ("start evaluating...---------------------------------------")
            
            self.eval.evaluate_agents()
            
            print ("start ranking Agents...-----------------------------------")

            self.learners, self.episode_rewards =  self.eval.rank_agents()
            
            for i in range(len(self.eval.eval_scores)):
                print('Average Reward for agent: ', i, '\n', np.average(self.eval.eval_scores[i]))
            for i in self.learners:
                print ('id:', i.id, 'ep_count:', i.ep_count,'ep:', i.episodes, 'learning rate:', i.agent.learning_rate)

            print ("start PBT process...---------------------------------------")
            self.exploitation(self.learners, self.episode_rewards)
            
            self.exploration(self.learners)

            for p in self.process_list:
                p.join()
                print(p)

            for i in self.learners:
                print ('id:', i.id, 'ep_count:', i.ep_count,'ep:', i.episodes)
            print (all(item.ep_count != item.episodes for item in self.learners))

        print ("finish training")
            
        # p = torch.multiprocessing.Pool()
        # p.map(self.run_learner, range(len(self.learners)))
        # p.close()
        # p.join()
        

    #Run inidivdual learners. Used as the target function for multiprocessing
    def run_learner(self, learner_idx):
        Admin.add_learner_profile(self.learners[learner_idx])
        
        # print ("learning rate", learner_idx, self.learners[learner_idx].agent.learning_rate)
        Admin.update_agents_profile(self.learners[learner_idx])
        self.learners[learner_idx].run()
        # print("episode",self.learners[learner_idx].ep_count)

    #fill the list of learners
    def populate_learners(self):
        for learner in range(self.learner_list_size):
            self.learners[learner] = self.create_learner()
            self.learners[learner].launch()
            self.sample(self.learners[learner])

    #Conduct Welch's T-test between two agents episode rewards and return whether we reject the null hypothesis
    def welch_T_Test(self,episodes_1, episodes_2):
        t,p = stats.ttest_ind(episodes_1, episodes_2, equal_var=False)

        return p < self.configs['MetaLearner']['p_value']

    def truncation(self,learners):
        truncate_size = int(self.learner_list_size* .2)
        bottom_20 = learners[self.learner_list_size - truncate_size:]
        top_20 = learners[:truncate_size]

        for learner in bottom_20:
            random_top_20 = random.choice(learners)
            learner.agent.policy = copy.deepcopy(random_top_20.agent.policy)
            learner.agent.optimizer = copy.deepcopy(random_top_20.agent.optimizer)
            learner.agent.learning_rate = copy.deepcopy(random_top_20.agent.learning_rate)

    def t_Test(self,learners,episode_rewards):
        for i in range(len(learners)):
            idxs = list(range(0,len(learners)))
            sampled_idx = random.choice(idxs)

            if(self.welch_T_Test(episode_rewards[i],episode_rewards[sampled_idx])):
                learners[i].agent.policy = copy.deepcopy(learners[sampled_idx].agent.policy)
                learners[i].agent.optimizer = copy.deepcopy(learners[sampled_idx].agent.optimizer)
                learners[i].agent.learning_rate = copy.deepcopy(learners[sampled_idx].agent.learning_rate)

    #randomly sample hyperparameter
    def sample (self, learner):
        new_lr = random.uniform(self.learning_rate_range[0],self.learning_rate_range[1])
        learner.agent.learning_rate = new_lr

    #Resample hyperparameters from the original range dictated in the configuration file
    def resample(self, learners):
        for learner in learners:
            new_lr = random.uniform(self.learning_rate_range[0],self.learning_rate_range[1])
            learner.agent.optimizer = getattr(torch.optim,self.configs['Agent']['optimizer_function'])(params=learner.agent.policy.parameters(), lr=new_lr)
            learner.agent.learning_rate = new_lr

    #Perturbate hyperparameters by a random factor randomly selected from predefined factor lists in the configuration file
    def perturbation(self,learners):
        for learner in learners:
            perturbation_factor = random.choice(self.configs['MetaLearner']['perturbation_factors'])
            new_lr = perturbation_factor * learner.agent.learning_rate
            learner.agent.optimizer = getattr(torch.optim,self.configs['Agent']['optimizer_function'])(params=learner.agent.policy.parameters(), lr=new_lr)
            learner.agent.learning_rate = new_lr

    def exploitation(self, learners, episode_rewards):
        if self.configs['MetaLearner']['exploit'] == 't_Test':
            self.t_Test(learners,episode_rewards)
        elif self.configs['MetaLearner']['exploit'] =='truncation':
            self.truncation(learners)

    def exploration(self,learners):
        if self.configs['MetaLearner']['explore'] == 'perturbation':
            self.perturbation(learners)
        elif self.configs['MetaLearner']['explore'] == 'resample':
            self.resample(learners)
