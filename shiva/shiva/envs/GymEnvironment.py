import gym
import numpy as np
from torch.distributions import Categorical
from shiva.envs.Environment import Environment
from shiva.helpers.misc import action2one_hot
import torch

class GymEnvironment(Environment):
    """ Gym Wrapper for interfacing with OpenAI Gym Environments.
        
        Args:
            configs (dict): Dictionary of environment hyperparameters, including
                the env_name, what device to run the class on, whether to render
                and reward modifiers. 
    
    """
    def __init__(self, configs:dict, *args, **kwargs):
        super(GymEnvironment, self).__init__(configs)

        self.env = gym.make(self.env_name)
        self.env.seed(self.manual_seed)
        np.random.seed(self.manual_seed)
        torch.manual_seed(self.manual_seed)

        # Set some attribute for Gym on MPI
        self.num_agents = 1
        self.roles = [self.env_name]
        self.num_instances_per_role = 1
        self.num_instances_per_env = 1

        self.action_space = {self.roles[0]: self.get_gym_action_space()}
        self.observation_space = {self.roles[0]: self.get_gym_observation_space()}
        self.reset()

        self.temp_done_counter = 0

    def step(self, action:list, discrete_select:str='argmax'):
        """ Takes an action in the environment.

        Note: 
            Every environment wrapper will have a step function that allows Shiva to interact with the environment. When
            the environment is discrete, it expects an integer as the action where as when the environment is continuous it 
            is expecting a float.

        Args:
            action (list): A list or tensor of probabilities or one hot encoding indicating which of the 
                index of the tensor should be used as the action to step on. The action is either a one 
                hot encoding or list of probabilties depending on how the algorithm was implemented.

            discrete_select (str): This dictates how the index of the action it the list is chosen. The
                action can be selected by either taking the 'argmax', or by 'sample'
        
        Returns:
            
            (np.array,int,bool,dict)

            Next Observation, Reward, Done Flag, and a Dictionary containing the raw Reward and Action stepped on.
        """
        if not torch.is_tensor(action):
            action = torch.tensor(action)
        self.acs = action

        if self.is_action_space_discrete():
            """Discrete, argmax or sample from distribution"""
            if not torch.is_tensor(action):
                action = torch.from_numpy(action)
            if discrete_select == 'argmax':
                action4Gym = torch.argmax(action).item()
            elif discrete_select == 'sample':
                action4Gym = Categorical(action).sample().item()
            self.obs, self.reward_per_step, self.done, info = self.env.step(action4Gym)
        else:
            """Continuous actions"""
            # self.obs, self.reward_per_step, self.done, info = self.env.step([action[action4Gym.item()]])
            self.obs, self.reward_per_step, self.done, info = self.env.step(action)

        """If Observation Space is discrete, turn it into a one-hot encode"""
        self.obs = self.transform_observation_space(self.obs)

        self.rew = self.normalize_reward(self.reward_per_step) if self.normalize else self.reward_per_step

        self.load_viewer()
        '''
            Metrics collection
                Episodic # of steps             self.steps_per_episode --> is equal to the amount of instances on Unity, 1 Shiva step could be a couple of Unity steps
                Cumulative # of steps           self.step_count
                Cumulative # of episodes        self.done_count
                Step Reward                     self.reward_per_step
                Episodic Reward                 self.reward_per_episode
                Cumulative Reward               self.reward_total
        '''
        self.steps_per_episode += 1
        self.step_count += 1
        self.done_count += 1 if self.done else 0
        self.reward_per_episode += self.rew
        self.reward_total += self.rew

        return self.obs, self.rew, self.done, {'raw_reward': self.reward_per_step, 'action': action}

    def reset(self, *args, **kwargs):
        """Reset episode metrics at the end of an episode or trajectory."""
        self.steps_per_episode = 0
        self.reward_per_step = 0
        self.reward_per_episode = 0
        self.done = False
        self.obs = self.transform_observation_space(self.env.reset())

    def get_metrics(self, episodic):
        """ Retrieve environment metrics.

        Args:
            episodic (bool): Indicates whether we're getting a stepwise or episodic metrics.
        
        Note:
            This is utilized by Shiva Admin in order to generate the plots in Tensorboard.

        Returns: 
            (str, int)

            A list of tuple metrics of the form (Metric_Name, Metric_Value)
        """
        if not episodic:
            metrics = [
                ('Reward/Per_Step', self.reward_per_step),
            ]
        else:
            metrics = [
                ('Reward/Per_Episode', self.reward_per_episode),
                ('Agent/Steps_Per_Episode', self.steps_per_episode)
            ]
        return [metrics] # single role metrics!
        # return metrics

    def is_done(self):
        """Check if the episode/trajectory has completed.
        
        Returns:
            (boolean)

            A done flag indicating whether the episode is over.
        """
        return self.done

    def transform_observation_space(self, raw_obs:np.array):
        """Converts a discrete observation into a One Hot Encoding.
        
        Returns: 
            (np.array) 

            One Hot Encoding if its a discrete action, otherwise it returns itself.
        """
        if self.is_observation_space_discrete():
            _one_hot_obs = np.zeros(self.observation_space)
            _one_hot_obs[raw_obs] = 1
            return _one_hot_obs
        else:
            return raw_obs

    def is_observation_space_discrete(self):
        """ Checks if the numpy shape of the observation space is scalar.
        
        If the observation space is a number it will return ().
        If the observation space is 2 dimensional or greater it will return a tuple
        with the corresponding dimensions inside.

        Returns: 
            (Boolean) 

        >>> import numpy as np
        >>> np.shape(1) # Discrete
        ()
        >>> np.shape(np.array([1,2]))
        (2,)
        """
        return self.env.observation_space.shape == ()

    def get_gym_observation_space(self):
        """ Calculates the observation space.

        If the environment is discrete then it can access the observation space but if its
        continuous then it has be calculated by multipling the dimensions. In short, it retrieves
        the observation space.

        Note:
            This is used internally by this wrapper.

        Returns: 
            (int)
        
        """
        observation_space = 1
        # if self.env.observation_space.shape != ():
        if not self.is_observation_space_discrete():
            # Continuous
            for i in range(len(self.env.observation_space.shape)):
                observation_space *= self.env.observation_space.shape[i]
        else:
            # Discrete
            observation_space = self.env.observation_space.n
        return observation_space

    def is_action_space_discrete(self):
        """ Checks if the numpy shape of the action space is scalar.
        
        If the action space is a number it will return ().
        If the action space is 2 dimensional or greater it will return a tuple
        with the corresponding dimensions inside.

        Returns: 
            (Boolean) 

        >>> import numpy as np
        >>> np.shape(1) # Discrete
        ()
        >>> np.shape(np.array([1,2]))
        (2,)
        """        
        return self.env.action_space.shape == ()

    def get_gym_action_space(self):
        """ Returns a dictionary 
        
        Returns: 
            (dict)
        """
        if self.is_action_space_discrete():
            return {
                'discrete': (self.env.action_space.n,),
                'continuous': 0,
                'param': 0,
                'acs_space': (self.env.action_space.n,),
                'actions_range': []
            }
        else:
            return {
                'discrete': 0,
                'continuous': self.env.action_space.shape,
                'param': 0,
                'acs_space': self.env.action_space.shape,
                'actions_range': [self.env.action_space.low, self.env.action_space.high]
            }

    def get_observations(self):
        """Returns a numpy array of numerical observations.

        Returns: 
            (np.array)
        """
        return self.obs

    def get_observation(self):
        """Returns a numpy array of numerical observations.
        
        Returns: 
            (np.array)    
        """
        return self.obs

    def get_actions(self):
        """ Returns stepwise actions.

        Returns: 
            (np.array)
        """
        return self.acs

    def get_action(self):
        """ Returns the stepwise action.

        Returns: 
            (np.array)
        """
        return self.acs

    def get_reward(self):
        """ Returns the stepwise reward.

        Returns: 
            (tensor)
        """
        return torch.tensor(self.reward_per_step)

    def get_total_reward(self):
        """Returns episodic reward

        Returns: 
            (int)
        """
        return self.reward_per_episode

    def get_reward_episode(self, roles:bool=False):
        """ Returns the episodic reward.

        Returns: 
            (dict) 
            
            A dictionary with the total reward indexed by the environment name.
        """
        if roles:
            return {self.roles[0]:self.reward_per_episode}
        return self.reward_per_episode

    def load_viewer(self):
        """ Shows the environment running.

        If you set render=True inside the config under the environment section it will make the
        environment visible during experiments.

        Note:
            Although its useful to see how the autonomous agent is behaving it is computationally more
            expensive.
        
        """
        if self.render:
            self.env.render()

    def close(self):
        """ Closes the environment instance.

        Once the preset number of episodes have ran the environment will close. Also there is a killswitch in case
        something causes Shiva to crash.
        """
        self.env.close()
