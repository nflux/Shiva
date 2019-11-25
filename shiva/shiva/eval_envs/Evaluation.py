class Evaluation(object):
    def __init__(self,
                configs: 'whole config passed'
    ):
        {setattr(self, k, v) for k,v in configs.items()}
        self.config = configs
        self._create_eval_envs()

    def evaluate_agents(self):
        '''
            Starts evaluation process
            This implementation is specific to each environment type
        '''
        pass

    def _create_eval_envs(self):
        '''
            This implementation is specific to each environment type
        '''
        pass

    def _start_evals(self):
        '''
            This implementation is specific to each environment type
        '''
        pass

    def rank_agents(self, validation_scores):
        pass
