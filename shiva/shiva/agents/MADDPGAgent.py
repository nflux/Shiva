import torch
from shiva.agents.DDPGAgent import DDPGAgent

class MADDPGAgent(DDPGAgent):
    def __init__(self, id: int, obs_space: int, acs_space: dict, agent_config: dict, networks: dict):
        super(MADDPGAgent, self).__init__(id, obs_space, acs_space, agent_config, networks)

    # def save(self, save_path, step):
    #     torch.save(self.actor, save_path + '/actor.pth')
    #     torch.save(self.target_actor, save_path + '/target_actor.pth')
    #     torch.save(self.critic, save_path + '/critic.pth')
    #     torch.save(self.target_critic, save_path + '/target_critic.pth')
    #     torch.save(self.actor_optimizer, save_path + '/actor_optimizer.pth')
    #     torch.save(self.critic_optimizer, save_path + '/critic_optimizer.pth')

    def get_module_and_classname(self):
        return ('shiva.agents', 'MADDPGAgent.MADDPGAgent')

    def __str__(self):
        return '<MADDPGAgent(id={}, role={}, steps={}, episodes={}, num_updates={}, device={})>'.format(self.id, self.role, self.step_count, self.done_count, self.num_updates, self.device)
