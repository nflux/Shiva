from shiva.core.communication_objects.env_command_pb2 import EnvironmentCommand
from shiva.core.communication_objects.env_step_pb2 import EnvStepInput, EnvStepOutput

def from_action_to_EnvStepInput(actions, command='step'):
    '''
        Need to set convention of actions shape with everybody
    '''
    env_in = EnvStepInput()

    fake_id = 0

    if len(actions.shape) == 1:
        # 1 Agent, 1 Instance
        action_obj = env_in.agent_actions[str(fake_id)].data.add()
        action_obj.data.extend(actions.tolist())
    if len(actions.shape) == 2:
        # 1 Agent, n Instances
        for instance_n in actions:
            action_obj = env_in.agent_actions[str(fake_id)].data.add()
            action_obj.data.extend(actions[instance_n])
    if len(actions.shape) == 3:
        # m Agents, n Instances
        for agent_m in range(actions.shape[0]):
            action_obj = env_in.agent_actions[str(fake_id)].data.add()
            for instance_n in range(actions.shape[1]):
                action_obj.data.extend(actions[agent_m, instance_n])
            fake_id += 1

    env_in.command = EnvironmentCommand.STEP
    return env_in

def from_EnvStepOutput_to_trajectories(env_output):
    '''
        This helper parses the EnvStepOutput into a dictionary @trajectories
    '''
    agent_ids = list(env_output.agent_states.keys())
    trajectories = {}
    for a_id in agent_ids:
        trajectories[a_id] = []
        for state in list(env_output.agent_states[a_id].data):
            trajectories[a_id].append([state.next_observation.data, state.reward, state.done, {}])
    return trajectories

def from_EnvStepOutput_to_metrics(env_output):
    agent_ids = list(env_output.agent_metrics.keys())
    agent_id = agent_ids[0]  # SINGLE AGENT PROCESSING
    metric = {}
    for a_id in agent_ids:
        metric[a_id] = env_output.agent_metrics[a_id].data[0]
    return metric