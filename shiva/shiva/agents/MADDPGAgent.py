from shiva.agents.DDPGAgent import DDPGAgent


class MADDPGAgent(DDPGAgent):
    """ MADDPG Agent Object

    Inherits from DDPG Agents for easier implementation.

    """

    epsilon = 0
    noise_scale = 0

    def __init__(self, id: int, obs_space: int, acs_space: dict, configs: dict):
        super(MADDPGAgent, self).__init__(id, obs_space, acs_space, configs)

    def get_metrics(self):
        """Gets the metrics so they can be passed to tensorboard.

        Returns:
             A tuple of metrics.
        """
        return [
            ('{}/Actor_Learning_Rate'.format(self.role), self.actor_learning_rate),
            # ('{}/Critic_Learning_Rate'.format(self.role), self.critic_learning_rate),
            ('{}/Epsilon'.format(self.role), self.epsilon),
            ('{}/Noise_Scale'.format(self.role), self.noise_scale),
        ]

    def get_module_and_classname(self):
        """ Returns the name of the module and class name.

        Returns:
            A string representation of the module and class.

        """
        return ('shiva.agents', 'MADDPGAgent.MADDPGAgent')

    def __str__(self):
        return f"<MADDPGAgent(id={self.id}, role={self.role}, S/E/U={self.step_count}/{self.done_count}/{self.num_updates}, T/P/R={self.num_evolutions['truncate']}/{self.num_evolutions['perturb']}/{self.num_evolutions['resample']} noise/epsilon={round(self.noise_scale, 2)}/{round(self.epsilon, 2)} device={self.device})>"