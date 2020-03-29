
from shiva.envs.RoboCupDDPGEnvironment import RoboCupDDPGEnvironment
from shiva.eval_envs.Evaluation import Evaluation

class RoboCupEvaluation(Evaluation):
    def __init__(self, configs, port):
        super(RoboCupEvaluation, self).__init__(configs)
        self.port = port
        
