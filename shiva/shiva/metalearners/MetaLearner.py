class MetaLearner(object):
    def __init__(self,
                learners : list, 
                algorithms : list, 
                eval_env : str, 
                agents : list, 
                elite_agents : list, 
                optimize_env_hp : bool, 
                optimize_learner_hp : bool, 
                evolution : bool,
                configs : list):
        self.learners = learners
        self.algorithms = algorithms
        self.eval_env = eval_env
        self.agents = agents
        self.elite_agents = elite_agents
        self.optimize_env_hp = optimize_env_hp
        self.optimize_learner_hp = optimize_env_hp
        self.evolution = evolution
        self.learnerCount = 0
        self.configs = configs
        
        self.PROD_MODE, self.EVAL_MODE = 'production', 'evaluation'
        self.mode = self.configs[0]['MetaLearner']['start_mode']

        env_name = self.configs[0]['Environment']['env_type'] + '-' + self.configs[0]['Environment']['environment']
        shiva.add_meta_profile(self, env_name)

    # this would play with different hyperparameters until it found the optimal ones
    def exploit_explore(self, hp, algorithms):
        pass

    def genetic_crossover(self,agents, elite_agents):
        pass

    def evolve(self, new_agents, new_hp):
        pass

    def evaluate(self, learners: list):
        pass

    def record_metrics(self):
        pass

    def id_generator(self):
        id = self.learnerCount
        self.learnerCount +=1
        return id

    def save(self):
        shiva.save(self)