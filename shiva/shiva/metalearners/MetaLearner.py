from settings import shiva

class MetaLearner(object):
    def __init__(self, algorithm, eval_env, agent, elite_agent, buffer, meta_config, learner_config, env_name):
        {setattr(self, k, v) for k,v in meta_config.items()}
        self.algorithm = algorithm
        self.eval_env = eval_env
        self.agent = agent
        self.elite_agent = elite_agent
        self.buffer = buffer
        self.meta_config = meta_config
        self.learner_config = learner_config
        self.learnerCount = 0
        
        self.PROD_MODE, self.EVAL_MODE = 'production', 'evaluation'

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

    def create_learner(self, learner_id, config):
        pass

    def id_generator(self):
        id = self.learnerCount
        self.learnerCount +=1
        return id

    def save(self):
        shiva.save(self)