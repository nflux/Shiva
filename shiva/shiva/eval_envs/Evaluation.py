class Evaluation(object):
    def __init__(self,
                eval_envs: 'list of environment names where to evaluate agents',
                learners: 'loaded leaners to be evaluated',
                metrics_strings: 'list of metric names to be recorded',
                render: 'rendering flag',
                config: 'whole config passed'
    ):
        self.eval_envs = eval_envs
        self.learners = learners
        self.metrics_strings = metrics_strings
        self.render = render
        self.config = config

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