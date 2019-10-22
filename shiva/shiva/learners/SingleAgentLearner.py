class SingleAgentLearner(Learner):
    def __init__(self, agents, environments, algorithm, data, configs):

        super(SingleAgentLearner,self).__init__(agents, environments, algorithm, data, configs)
        self.agents = agents
        self.environments = environments
        self.algorithm = algorithm
        self.data = None
        self.configs = configs


    def update(self):

        step_count = 0
        
        for ep_count in range(self.configs['Learner']['episodes']):
            self.env.reset()
            self.totalReward = 0
            done = False
            while not done:
                done = self.step(step_count, ep_count)
                step_count += 1

        self.env.env.close()


    def step(self, step_count, ep_count):

        # self.env.env.render()

        observation = self.env.get_observation()

        action = self.alg.get_action(self.alg.agents[0], observation, step_count)

        next_observation, reward, done = self.env.step(action)

        # TensorBoard metrics
        self.writer.add_scalar('Actor Loss per Step', self.alg.get_actor_loss(), step_count)
        self.writer.add_scalar('Critic Loss per Step', self.alg.get_critic_loss(), step_count)
        self.writer.add_scalar('Reward', reward, step_count)
        self.totalReward += reward

        self.buffer.append([observation, action, reward, next_observation, int(done)])

        if step_count > 0:
            self.agents = self.alg.update(self.agents, self.buffer.sample(), step_count)

        # TensorBoard Metrics
        if done:
            self.writer.add_scalar('Total Reward', self.totalReward, ep_count)
            self.alg.ou_noise.reset()

        return done

    def create_environment(self):
        # create the environment and get the action and observation spaces
        self.env = Environment.initialize_env(self.configs['Environment'])


    def get_agents(self):   
        return self.agents[0]

    def get_algorithm(self):
        return self.algorithm

    def launch(self):

        # Launch the environment
        self.create_environment()

        # Launch the algorithm which will handle the
        self.alg = Algorithm.initialize_algorithm(self.env.get_observation_space(), self.env.get_action_space(), [self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])

        self.agents = self.alg.create_agent()

        self.writer = SummaryWriter()

        # Basic replay buffer at the moment
        self.buffer = ReplayBuffer.initialize_buffer(self.configs['Replay_Buffer'], 1, self.env.get_action_space(), self.env.get_observation_space())

        print('Launch done.')


    def save_agent(self):
        pass

    def load_agent(self):
        pass