import Environment

def initialize_eval_env(env_params):
    assert env_params)
    assert 'env_type' in env_params: 'Needs @env_type in [EVAL_ENV] config'

    if env_params['env_type'] == 'Gym':
        return GymEvaluationEnvironment(env_params['environment'], env_params['env_render'])

class GymEvaluationEnvironment(Environment.GymEnvironment)
    def __init__(self, environment, render):
        super(GymEvaluationEnvironment, self).__init__(environment, render)
        pass

    def get_metrics(self):
        '''
            Return
                List of metric tuples
                g.e. [ ('cumulative_reward', 100), (...), (...) ]
        '''
        pass