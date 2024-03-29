import EvaluationEnvironment

def initialize_evaluation(config):
    # print(config)
    if config['env_type'] == 'Gym':
        return GymEvaluation(config['environment'], config['learners'], config['metrics'], config['env_render'], config)

class AbstractEvaluation(object):
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


class GymEvaluation(AbstractEvaluation):
    def __init__(self, 
            eval_envs,
            learners,
            metrics_strings,
            render,
            config
        ):
        super(GymEvaluation, self).__init__(eval_envs, learners, metrics_strings, render, config)
        self.agent_metrics = {}

    def _create_eval_envs(self):
        '''
            Initializes the evaluation environments
            It's executed at initialization by the AbstractEvaluation, but the implementation is specific to each Environment
        '''
        self.eval_envs = [ EvaluationEnvironment.initialize_eval_env({'env_type': 'Gym', 'environment': env_name, 'metrics_strings': self.metrics_strings, 'env_render': self.render}) for env_name in self.eval_envs ]

    def evaluate_agents(self):
        '''
            Starts the evaluation process of each agent on each environment
        '''
        for env in self.eval_envs:
            self.agent_metrics[env.env_name] = {}
            for learner in self.learners:
                self.agent_metrics[env.env_name][learner.id] = {}
                for agent in learner.agents:
                    print("Start evaluation for {} on {}".format(agent, env))
                    self._make_agent_play(env, agent)
                    self.agent_metrics[env.env_name][learner.id][agent.id] = env.get_metrics()
                    print("Finish evaluation for {}".format(agent))

    def _make_agent_play(self, env, agent):
        '''
            Let's the agent play for X episodes given by Evaluation config
        '''
        for ep_n in range(self.config['episodes']):
            # print("Episode {}".format(ep_n))
            env.reset()
            done = False
            while not done:
                done = self._step_agent(env, agent)
        env.close()

    def _step_agent(self, env, agent):
        '''
            Literally makes the agent do 1 step on the environment
        '''
        observation = env.get_observation()
        action = agent.get_action(observation)
        next_observation, reward, done = env.step(action)
        return done