import torch
import random, dill
from scipy import stats
import copy
import numpy as np

from shiva.core.admin import Admin
from shiva.metalearners.MetaLearner import MetaLearner
from shiva.helpers.config_handler import load_class

class SingleAgentPBTMetaLearner(MetaLearner):
    def __init__(self, configs):
        super(SingleAgentPBTMetaLearner, self).__init__(configs)
        self.learners = [None] * self.num_learners
        self.evals = [None] * self.num_learners
        self.process_list = []
        if hasattr(self, 'start_port'):
            # 7 denotes the number of ports needed per imitation learner
            # 4 per bot env and 3 per learner env
            self.learner_ports = [(self.start_port + (7*i)) for i in range(self.num_learners)]
            # 3 denotes the number of ports needed per eval_env
            self.eval_ports = [((self.learner_ports[-1] + 7) + (3*i)) for i in range(self.num_learners)]

        self.learner_processes = []
        self.eval_processes = []
        self.run()
    
    def run(self):
        torch.multiprocessing.set_start_method('spawn')
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
            try:
                self.populate_objs()
                print('Populated Objects')
                self.process_learners()
            except KeyboardInterrupt:
                print('Exiting for CTRL-C')
            finally:
                print('Cleaning up possible extra learner processes')
                [l.close() for l in self.learners]
                [p.join() for p in self.learner_processes]
                
        
        # for i in range(len(self.eval.eval_scores)):
        #     print('Average Reward for agent: ', i, '\n', np.average(self.eval.eval_scores[i]))
        self.save()
        print('bye')
    
    def process_learners(self):
        for l in self.learners:
            s = "learner_" + str(l)
            pick_run = dill.dumps(l.run)
            p = torch.multiprocessing.Process(name=s, target=dill.loads(pick_run))
            p.start()
            print('Passssed hereerereer')
            self.learner_processes.append(p)

        eval_check = [False] * self.num_learners
        not_finished = lambda x: any([i.ep_count < i.episodes for i in x])
        while not_finished(self.learners):
            for l in self.learners:
                if (l.ep_count % self.eps_per_eval) == 0:
                    l.train[0] = False
                    eval_check[l.id] = True
            if all(eval_check):
                self.eval_processes = []
                for e in self.evals:
                    s = "eval_" + str(e)
                    p = torch.multiprocessing.Process(name=s, target=e.launch)
                    p.start()
                    self.eval_processes.append(p)
                
                for i,p in enumerate(self.eval_processes):
                    p.join()
                    eval_check[i] = False
                    self.learners[i].train[0] = True

                print([e.score for e in self.evals])

    #Run inidivdual learners. Used as the target function for multiprocessing
    # def run_learner(self, learner_idx):
    #     self.learners[learner_idx].run()
    
    # def run_eval(self, eval_idx):
    #     self.evals[eval_idx].launch()
    
    def create_learner(self, idx):
        learner_class = load_class('shiva.learners', self.configs['Learner']['type'])
        return learner_class(self.get_id(), self.configs, self.learner_ports[idx])
    
    def create_eval(self, agent, idx):
        eval_class = load_class('shiva.eval_envs', self.configs['Evaluation']['type'])
        return eval_class(self.configs, agent, self.eval_ports[idx])

    # fill the list of learners and eval_envs
    def populate_objs(self):
        for idx in range(self.num_learners):
            self.learners[idx] = self.create_learner(idx)
            Admin.add_learner_profile(self.learners[idx])
            self.learners[idx].launch()
            Admin.update_agents_profile(self.learners[idx])
            self.sample(self.learners[idx])

            self.evals[idx] = self.create_eval(self.learners[idx].get_agent(), idx)
    
    def update_evals(self):
        for e,l in zip(self.evals, self.learners):
            e.agent = l.get_agent()

    def evaluate(self):
        processes = []
        self.update_evals()
        for e in self.evals:
            p = torch.multiprocessing.Process(target=e.launch())
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        return sorted([(i, e.score) for i,e in enumerate(self.evals)], key=lambda e: e[1])

    #Conduct Welch's T-test between two agents episode rewards and return whether we reject the null hypothesis
    def welch_T_Test(self,episodes_1, episodes_2):
        t, p = stats.ttest_ind(episodes_1, episodes_2, equal_var=False)
        return p < self.p_value

    def truncation(self, learners):
        truncate_size = int(self.learner_list_size* .2)
        bottom_20 = learners[self.learner_list_size - truncate_size:]
        top_20 = learners[:truncate_size]

        for learner in bottom_20:
            random_top_20 = random.choice(top_20)
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
        new_lr = random.uniform(self.learning_rate_range[0], self.learning_rate_range[1])
        learner.agent.learning_rate = new_lr

    #Resample hyperparameters from the original range dictated in the configuration file
    def resample(self, learners):
        resample_size = int(self.learner_list_size* .2)
        bottom_20 = learners[self.learner_list_size - resample_size:]
        for learner in bottom_20:
            new_lr = random.uniform(self.learning_rate_range[0],self.learning_rate_range[1])
            # lmbda = lambda epoch: new_lr
            # scheduler = LambdaLR(learner.agent.optimizer, lr_lambda=lmbda)
            # scheduler.step()
            learner.agent.optimizer = getattr(torch.optim, self.configs['Agent']['optimizer_function'])(params=learner.agent.policy.parameters(), lr=new_lr)
            learner.agent.learning_rate = new_lr

    #Perturbate hyperparameters by a random factor randomly selected from predefined factor lists in the configuration file
    def perturbation(self,learners):
        perturbation_size = int(self.learner_list_size* .2)
        bottom_20 = learners[self.learner_list_size - perturbation_size:]
        for learner in bottom_20:
            perturbation_factor = random.choice(self.perturbation_factors)
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