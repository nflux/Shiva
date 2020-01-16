from shiva.envs.RoboCupDDPGEnvironment import RoboCupDDPGEnvironment
from shiva.eval_envs.Evaluation import Evaluation

class RoboCupEvaluation(Evaluation):
    def __init__(self, configs, agent, port):
        super(RoboCupEvaluation, self).__init__(configs)
        self.port = port
        self.agent = agent
        self.env = RoboCupDDPGEnvironment(configs['Evaluation'], port)
        self.score = 0.0

    def launch(self):
        self.env.launch()
        self.evaluate_agent()

    def evaluate_agent(self):
        for e in range(self.episodes):
            done = False
            while not done:
                done = self.step()
        
        # Only get goal percentage for now
        self.score = self.env.get_metrics(episodic=True)[2][1]

    def step(self):
        observation = self.env.get_observation()
        action = self.agent.find_best_imitation_action(observation)
        _, _, done, _ = self.env.step(action)

        return done