from settings import shiva
from .Learner import Learner
import helpers.misc as misc
import envs
import algorithms
import buffers

class SingleAgentLearner(Learner):
    def __init__(self, learner_id, config):
        super(SingleAgentLearner,self).__init__(learner_id, config)


    def run(self):
        step_count = 0
        for ep_count in range(self.episodes):
            self.env.reset()
            self.totalReward = 0
            done = False
            while not done:
                step_count +=1
                done = self.step(step_count, ep_count)

        self.env.close()

    # def update(self):

    #     step_count = 0
        
    #     for ep_count in range(self.episodes):
    #         self.env.reset()
    #         self.totalReward = 0
    #         done = False
    #         while not done:
    #             done = self.step(step_count, ep_count)
    #             step_count += 1

    #     self.env.env.close()


    # def step(self, step_count, ep_count):

    #     # self.env.env.render()

    #     observation = self.env.get_observation()

    #     action = self.alg.get_action(self.alg.agent, observation, step_count)

    #     next_observation, reward, done = self.env.step(action)

    #     # TensorBoard metrics
    #     self.writer.add_scalar('Actor Loss per Step', self.alg.get_actor_loss(), step_count)
    #     self.writer.add_scalar('Critic Loss per Step', self.alg.get_critic_loss(), step_count)
    #     self.writer.add_scalar('Reward', reward, step_count)
    #     self.totalReward += reward

    #     self.buffer.append([observation, action, reward, next_observation, int(done)])

    #     if step_count > 0:
    #         self.agents = self.alg.update(self.agents, self.buffer.sample(), step_count)

    #     # TensorBoard Metrics
    #     if done:
    #         self.writer.add_scalar('Total Reward', self.totalReward, ep_count)
    #         self.alg.ou_noise.reset()

    #     return done

    # Function to step throught the environment
    def step(self, step_count, ep_count):

        self.env.load_viewer()

        observation = self.env.get_observation()

        action = self.alg.get_action(self.agent, observation, self.env.get_current_step())

        next_observation, reward, done = self.env.step(action)

        # Write to tensorboard
        shiva.add_summary_writer(self, self.agent, 'Reward', reward, step_count)
        shiva.add_summary_writer(self, self.agent, 'Loss per Step', self.alg.get_loss(), step_count)

        # Cumulate the reward
        self.totalReward += reward[0]

        self.buffer.append([observation, action, reward, next_observation, done])

        self.alg.update(self.agent, self.buffer.sample(), step_count)
        # when the episode ends
        if done:
            shiva.add_summary_writer(self, self.agent, 'Total Reward', self.totalReward, ep_count)

        # Save the agent periodically
        if step_count % self.save_frequency == 0:
            shiva._save_agent(self, self.agent)

        return done

    # def create_environment(self):
    #     # create the environment and get the action and observation spaces
    #     self.env = Environment.initialize_env(self.configs['Environment'])


    def get_agents(self):   
        return self.agents

    def get_algorithm(self):
        return self.alg

    def launch(self):

        # Launch the environment
        self.env = getattr(envs, self.configs['Environment']['type'])(self.configs['Environment'])

        # Launch the algorithm which will handle the
        self.alg = getattr(algorithms, self.configs['Algorithm']['type'])(self.env.get_observation_space(), self.env.get_action_space(),[self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])

        self.agent = self.alg.create_agent()

        # Basic replay buffer at the moment
        self.buffer = getattr(buffers,self.configs['Buffer']['type'])(self.configs['Buffer']['batch_size'], self.configs['Buffer']['capacity'])

        print('Launch done.')


    def save_agent(self):
        pass

    def load_agent(self, path):
        return shiva._load_agents(path)[0]
