class Meta_Learner():
    def __init__(self, learners, algorithms, eval_env, agents, elite_agents, optimize_env_hp=False, optimize_learner_hp=False, evolution=False):
        self.learners = learners
        self.algorithms = algorithms
        self.eval_env = eval_env
        self.agents = agents
        self.elite_agents = elite_agents
        self.optimize_env_hp = optimize_env_hp
        self.optimize_learner_hp = optimize_env_hp
        self. evolution = evolution

    # this would play with different hyperparameters until it found the optimal ones
    def exploit_explore(self, hp, algorithms):

        # there might be different algorithms that will tune the hyperparameters

        pass
        #return new_hp
    
    # these woud breed
    def genetic_crossover(self,agents, elite_agents):
        pass

        #return new_agents

    def evolve(self, new_agents, new_hp):

        self.algorithms
        pass

    def evaluate(self, learners):
        pass

    def record_metrics(self):
        pass

