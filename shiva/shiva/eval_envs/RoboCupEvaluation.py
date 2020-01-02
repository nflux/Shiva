from shiva.envs.RoboCupDDPGEnvironment import RoboCupDDPGEnvironment
from shiva.eval_envs.Evaluation import Evaluation

class RoboCupEvaluation(Evaluation):
    def __init__(self, configs, agent, port):
        super(RoboCupEvaluation, self).__init__(configs)
        self.port = port
        self.agent = agent
        self.env = RoboCupDDPGEnvironment(configs, port)
        self.goal_ctr = 0

    def evaluate_agent(self):
        self.goal_ctr = 0
        for e in range(self.config['episodes']):
            done = False
            while not done:
                done = self.step()
            if self.env.isGoal():
                self.goal_ctr += 1
        
        return (self.goal_ctr/self.config['episodes'])*100.0

    def step(self):
        observation = self.env.get_observation()
        action = self.agent.find_best_imitation_action(observation)
        _, _, done, _ = self.env.step(action)

        return done

        
