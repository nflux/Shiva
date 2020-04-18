import torch
from shiva.agents.DDPGAgent import DDPGAgent

class MADDPGAgent(DDPGAgent):
    def __init__(self, id: int, obs_space: int, acs_space: dict, agent_config: dict, networks: dict):
        super(MADDPGAgent, self).__init__(id, obs_space, acs_space, agent_config, networks)

    def get_metrics(self):
        '''Used for evolution metric'''
        return [
            ('{}/Actor_Learning_Rate'.format(self.role), self.actor_learning_rate),
            # ('{}/Critic_Learning_Rate'.format(self.role), self.critic_learning_rate),
            ('{}/Epsilon'.format(self.role), self.epsilon),
            ('{}/Noise_Scale'.format(self.role), self.noise_scale),
        ]

    def get_module_and_classname(self):
        return ('shiva.agents', 'MADDPGAgent.MADDPGAgent')

    def __str__(self):
        return '<MADDPGAgent(id={}, role={}, steps={}, episodes={}, num_updates={}, device={})>'.format(self.id, self.role, self.step_count, self.done_count, self.num_updates, self.device)
