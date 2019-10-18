import EvaluationEnvironment

def initialize_evaluation(config):
    print(config)
    if config['env_type'] == 'Gym':
        return GymEvaluation(config['environment'], config['learners'], config['metrics'], config['env_render'])

class AbstractEvaluation(object):
    def __init__(self,
                learners: list,
                eval_envs: list,
                eval_config: dict
    ):
        self.learners = learners
        self.eval_envs = eval_envs
        self.eval_config = eval_config

    def evaluate_agents(self,
                agents: list,
                envs_config: list):
        '''
            Input
                agents          list of agents to be evaluated
                envs_config     list of configs for environments to evaluate the agents
        '''
        self.agents = agents
        self.envs_config = envs_config
        self._create_eval_envs()
        self._start_evals()

    def _create_eval_envs(self):
        '''
            Creates self attributes for the environments where to evaluate the agents
        '''
        self.eval_envs = [EvaluationEnvironment.initialize_eval_env(config) for config in self.envs_config]

    def _start_evals(self):
        '''
            This implementation is specific to each environment
        '''
        pass

    def rank_agents(self, validation_scores):
        pass


class GymEvaluation(AbstractEvaluation):
    def __init__(self, 
            envs_strs: 'list of envs',
            agents: 'list of agents',
            metrics: 'list of metrics to calculate',
        ):
        super(GymEvaluation, self).__init__(envs_strs, agents, metrics)

    def _start_evals():
        '''
            Starts the evaluation process of each agent on each environment
                Maybe it doesn't make sense to have many envs here.... since it's a Gym
        '''
        for env in self.eval_envs:
            for agent in self.agents:
                self._make_agent_play(env, agent)

    def _make_agent_play(self, env, agent):
        '''
            Let's the agent play for X episodes
        '''
        for _ in range(self.eval_config['episodes']):
            env.reset()
            done = False
            while not done:
                done = self._step_agent(env, agent)
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

    