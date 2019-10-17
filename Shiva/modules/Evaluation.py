import EvaluationEnvironment

def initialize_evaluation(config):
    print(config)
    if config['env_type'] == 'Gym':
        return GymEvaluation(config['environment'], config['learners'], config['metrics'], config['env_render'], config)

class AbstractEvaluation(object):
    def __init__(self,
                eval_envs: 'list of environment names where to evaluate agents',
                learners: 'loaded leaners to be evaluated',
                metrics: 'list of metric names to be recorded',
                render: 'rendering flag',
                config: 'whole config passed'
    ):
        self.eval_envs = eval_envs
        self.learners = learners
        self.metrics = metrics
        self.render = render
        self.config = config

        self._create_eval_envs()

    def evaluate_agents(self):
        '''
            Starts evaluation process
        '''
        pass

    def _create_eval_envs(self):
        '''
            This may be specific to each environment
        '''
        pass
    
    def _start_evals(self):
        '''
            This implementation is specific to each environment
        '''
        pass

    def rank_agents(self, validation_scores):
        pass


class GymEvaluation(AbstractEvaluation):
    def __init__(self, 
            envs_strs,
            learners,
            metrics,
            render,
            config
        ):
        super(GymEvaluation, self).__init__(envs_strs, learners, metrics, render, config)

    def _create_eval_envs(self):
        self.eval_envs = [ EvaluationEnvironment.initialize_eval_env({'env_type': 'Gym', 'environment': env_name, 'env_render': self.render}) for env_name in self.eval_envs]

    def evaluate_agents(self):
        '''
            Starts the evaluation process of each agent on each environment
                Maybe it doesn't make sense to have many envs here.... since it's a Gym
        '''
        for env in self.eval_envs:
            for learner in self.learners:
                for agent in learner:
                    self._make_agent_play(env, agent)

    def _make_agent_play(self, env, agent):
        '''
            Let's the agent play for X episodes
        '''
        print("Agent {} start".format(agent.id))
        for _ in range(self.config['episodes']):
            env.reset()
            done = False
            while not done:
                done = self._step_agent(env, agent)
        print("Agent {} finish".format(agent.id))
        env.close()

    def _step_agent(self, env, agent):
        '''
            Literally makes the agent step on the environment
        '''
        env.load_viewer()
        observation = env.get_observation()
        action = agent.get_action(observation)
        next_observation, reward, done = env.step(action)
        # self.writer.add_scalar('Reward',reward, self.env.get_current_step())
        # self.buffer.append([observation, action, reward, next_observation, done])
        # if self.env.get_current_step() % self.saveFrequency == 0:
        #     pass
        return done

    