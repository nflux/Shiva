import json
from shiva.helpers.timers import timed

from shiva.core.communication_objects.specs_pb2 import (
    SpecsProto, ActionSpaceProto, EnvSpecsProto, MultiEnvSpecsProto, LearnerSpecsProto
)
from shiva.core.communication_objects.configs_pb2 import ConfigProto
from shiva.core.communication_objects.helpers_pb2 import SimpleMessage
from shiva.core.communication_objects.enums_pb2 import ComponentType

@timed
def from_dict_2_ConfigProto(configs: dict) -> ConfigProto:
    config_proto = ConfigProto()
    config_proto.data = json.dumps(configs)
    return config_proto

def from_ConfigProto_2_dict(config_proto: ConfigProto) -> dict:
    config = json.loads(config_proto.data)
    return config

@timed
def from_dict_2_ObservationsProto(observations):
    assert "NotImplemented"

@timed
def from_ObservationsProto_2_dict():
    assert "NotImplemented"

@timed
def from_dict_2_ActionsProto():
    assert "NotImplemented"

@timed
def from_ActionsProto_2_dict():
    assert "NotImplemented"

@timed
def from_dict_2_TrajectoriesProto():
    assert "NotImplemented"

@timed
def from_TrajectoriesProto_2_dict():
    assert "NotImplemented"

@timed
def from_dict_2_NewAgentsConfigProto():
    assert "NotImplemented"

@timed
def from_NewAgentsConfigProto_2_dict():
    assert "NotImplemented"

@timed
def from_dict_2_TrainingMetricsProto():
    assert "NotImplemented"

@timed
def from_EvolutionMetricProto_2_dict():
    assert "NotImplemented"

@timed
def from_dict_2_EvolutionMetricProto():
    assert "NotImplemented"

@timed
def from_TrainingMetricsProto_2_dict():
    assert "NotImplemented"

@timed
def from_dict_2_EvolutionConfigProto():
    assert "NotImplemented"

@timed
def from_EvolutionConfigProto_2_dict():
    assert "NotImplemented"

@timed
def from_dict_2_MultiEnvSpecsProto(menv_specs: dict) -> MultiEnvSpecsProto:
    menv_specs_proto = MultiEnvSpecsProto()
    menv_specs_proto.num_envs = menv_specs['num_envs']
    menv_specs_proto.env_specs.observation_space = menv_specs['env_specs']['observation_space']
    menv_specs_proto.env_specs.action_space.discrete = menv_specs['env_specs']['action_space']['discrete']
    menv_specs_proto.env_specs.action_space.param = menv_specs['env_specs']['action_space']['param]']
    menv_specs_proto.env_specs.action_space.acs_space = menv_specs['env_specs']['action_space']['acs_space']
    menv_specs_proto.env_specs.num_agents = menv_specs['env_specs']['num_agents']
    return menv_specs_proto

@timed
def from_MultiEnvSpecsProto_2_dict(menv_specs_proto):
    menv_specs = {}
    menv_specs['num_envs'] = menv_specs.num_envs
    menv_specs['env_specs'] = from_EnvSpecsProto_to_dict(menv_specs.env_specs)
    return menv_specs

def from_dict_2_ActionSpaceProto(action_space: dict) -> ActionSpaceProto:
    action_space_proto = ActionSpaceProto()
    action_space_proto.discrete = action_space['discrete']
    action_space_proto.param = action_space['param']
    action_space_proto.acs_space = action_space['acs_space']
    return action_space_proto

def from_ActionSpaceProto_2_dict(action_space_proto: dict) -> dict:
    action_space = {}
    action_space['discrete'] = action_space_proto.discrete
    action_space['param'] = action_space_proto.param
    action_space['acs_space'] = action_space_proto.acs_space
    return action_space

@timed
def from_dict_2_EnvSpecsProto(env_specs: dict):
    env_specs_proto = EnvSpecsProto()
    env_specs_proto.observation_space = env_specs['observation_space']
    env_specs_proto.action_space.discrete = env_specs['action_space']['discrete']
    env_specs_proto.action_space.param = env_specs['action_space']['param']
    env_specs_proto.action_space.acs_space = env_specs['action_space']['acs_space']
    env_specs_proto.num_agents = env_specs['num_agents']
    return env_specs_proto

@timed
def from_EnvSpecsProto_to_dict(env_specs_proto: EnvSpecsProto) -> dict:
    env_specs = {}
    env_specs['observation_space'] = env_specs_proto.observation_space
    env_specs['action_space'] = from_ActionSpaceProto_2_dict(env_specs_proto.action_space)
    env_specs['num_agents'] = env_specs_proto.num_agents
    return env_specs

@timed
def from_LearnerSpecsProto_2_dict(learner_specs_proto: LearnerSpecsProto) -> dict:
    learner_specs = {}
    learner_specs['data'] = json.load(learner_specs_proto.data)
    return learner_specs

@timed
def from_dict_2_SpecsProto(specs: dict) -> SpecsProto:
    specs_proto = SpecsProto()
    specs_proto.id = specs['id']
    specs_proto.type = specs['type']
    if 'address' in specs:
        specs_proto.address = specs['address']
    if specs['type'] == ComponentType.LEARNER:
        specs_proto.learner.data = json.dumps(specs['data'])
    elif specs['type'] == ComponentType.MULTIENV:
        specs_proto.menv.num_envs = specs['num_envs']
        specs_proto.menv.env_specs.observation_space = specs['env_specs']['observation_space']
        specs_proto.menv.env_specs.action_space.discrete = specs['env_specs']['action_space']['discrete']
        specs_proto.menv.env_specs.action_space.param = specs['env_specs']['action_space']['param']
        specs_proto.menv.env_specs.action_space.acs_space = specs['env_specs']['action_space']['acs_space']
    elif specs['type'] == ComponentType.ENVIRONMENT:
        pass
    return specs_proto

@timed
def from_SpecsProto_2_dict(specs_proto: SpecsProto) -> dict:
    specs = {}
    specs['id'] = specs_proto.id
    specs['type'] = specs_proto.type
    if specs_proto.type == ComponentType.LEARNER:
        specs['learner'] = from_LearnerSpecsProto_2_dict(specs_proto.learner)
    elif specs_proto.type == ComponentType.MULTIENV:
        specs['menv'] = from_MultiEnvSpecsProto_2_dict(specs_proto.menv)
    elif specs_proto.type == ComponentType.ENVIRONMENT:
        specs['env'] = from_EnvSpecsProto_to_dict(specs_proto.env)
    return specs

@timed
def from_dict_2_JsonMessage(msg) -> JsonMessage:
    simple = JsonMessage()
    simple.data = json.dumps(msg)
    return simple

@timed
def from_JsonMessage_2_dict(simple_msg: JsonMessage) -> dict:
    return json.load(simple_msg.data)

@timed
def from_SimpleMessage_2_int(simple_msg_proto: SimpleMessage) -> int:
    assert "Implementation not checked"
    return int(simple_msg_proto.data)

@timed
def from_SimpleMessage_2_string(simple_msg_proto: SimpleMessage) -> str:
    assert "Implementation not checked"
    return str(simple_msg_proto.data)

@timed
def from_dict_2_StatusProto():
    assert "NotImplemented"

@timed
def from_StatusProto_2_dict():
    assert "NotImplemented"




#
#
#
#
#
#

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