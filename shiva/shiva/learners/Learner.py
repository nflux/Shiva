import csv
import datetime
import time
# import os.path
from shiva.core.admin import Admin
from shiva.helpers.config_handler import load_class

class Learner(object):
    
    firstRun = False
    def __init__(self, learner_id, config):
        {setattr(self, k, v) for k,v in config['Learner'].items()}
        self.configs = config
        self.environmentName = str(self.configs['Environment']['env_name'])
        self.algType = str(self.configs['Algorithm']['Type'])
        # print('Hello World' + self.env)
        self.id = learner_id
        self.agentCount = 0
        self.ep_count = 0
        self.step_count = 0
        self.checkpoints_made = 0
        self.firstRun = False
        ts = time.time()
        tS = str(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
        self.timeStamp = tS
        
    def __getstate__(self):
        d = dict(self.__dict__)
        try:
            del d['env']
        except KeyError:
            del d['envs']
        return d
        try:
            del d['eval']
        except KeyError:
            pass
        return d

    def collect_metrics(self, episodic=False):
        '''
            This works for Single Agent Learner
            For Multi Agent Learner we need to implemenet the else statement
        '''
        
        
        if hasattr(self, 'agent') and type(self.agent) is not list:
            metrics = self.alg.get_metrics(episodic) + self.env.get_metrics(episodic)
            if not episodic:
                for metric_name, y_val in metrics:
                    Admin.add_summary_writer(self, self.agent, metric_name, y_val, self.env.step_count)
                    try:
                        if (self.firstRun == False):
                            file = open("Benchmark/"+str(metric_name)+" " +self.algType+" "+ self.environmentName +" "+ self.timeStamp+'.csv', 'w+')
                            file.close()
                            self.firstRun = True
                        f = open("Benchmark/"+str(metric_name)+" " +self.algType+" "+ self.environmentName +" "+ self.timeStamp+'.csv')
                        f.close()
                    except FileNotFoundError:
                        file = open("Benchmark/"+str(metric_name)+" " +self.algType+" "+ self.environmentName +" "+ self.timeStamp+'.csv', 'w+')
                        file.close()
                    
                    with open("Benchmark/"+str(metric_name)+" " +self.algType+" "+ self.environmentName +" "+ self.timeStamp+'.csv', 'a', newline='') as csvfile:
                        fieldnames = ['steps','rewards']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow({'steps': self.env.step_count, 'rewards': str(y_val)})
            else:
                for metric_name, y_val in metrics:
                    Admin.add_summary_writer(self, self.agent, metric_name, y_val, self.env.done_count)

                    try:
                        if (self.firstRun == False):
                            file = open("Benchmark/"+str(metric_name)+" " +self.algType+" "+ self.environmentName +" "+ self.timeStamp+'.csv', 'w+')
                            file.close()
                            self.firstRun = True
                        f = open("Benchmark/"+str(metric_name)+" " +self.algType+" "+ self.environmentName +" "+ self.timeStamp+'.csv')
                        f.close()
                    except FileNotFoundError:
                        file = open("Benchmark/"+str(metric_name)+" " +self.algType+" "+ self.environmentName +" "+ self.timeStamp+'.csv', 'w+')
                        file.close()
                    with open("Benchmark/"+str(metric_name)+" " +self.algType+" "+ self.environmentName +" "+ self.timeStamp+'.csv', 'a', newline='') as csvfile:
                        fieldnames = ['steps','rewards']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow({'steps': self.env.step_count, 'rewards': str(y_val)})
        else:
            assert False, "The Learner attribute 'agent' was not found. Either name the attribute 'agent' or could be that MultiAgent Metrics are not yet supported."
        

    def checkpoint(self):
        assert hasattr(self, 'save_checkpoint_episodes'), "Learner needs 'save_checkpoint_episodes' attribute in config - put 0 if don't want to save checkpoints"
        if self.save_checkpoint_episodes > 0:
            t = self.save_checkpoint_episodes * self.checkpoints_made
            if self.env.done_count > t:
                print("%% Saving checkpoint at episode {} %%".format(self.env.done_count))
                Admin.update_agents_profile(self)
                self.checkpoints_made += 1

    def update(self):
        assert 'Not implemented'
        pass

    def step(self):
        assert 'Not implemented'
        pass

    def create_environment(self):
        env_class = load_class('shiva.envs', self.configs['Environment']['type'])
        return env_class(self.configs['Environment'])

    def get_agents(self):
        assert 'Not implemented'
        pass

    def get_algorithm(self):
        assert 'Not implemented'
        pass

    def launch(self):
        assert 'Not implemented'
        pass

    def save(self):
        Admin.save(self)

    def load(self, attrs):
        for key in attrs:
            setattr(self, key, attrs[key])

    def get_id(self):
        id = self.agentCount
        self.agentCount +=1
        return id