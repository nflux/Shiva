from shiva.helpers.timers import timed

from shiva.core.communication_objects.env_step_pb2 import EnvironmentCommand, ObservationsProto, ActionsProto, TrajectoriesProto
from shiva.core.communication_objects.configs_pb2 import ConfigProto, StatusProto
# from shiva.core.communication_objects.specs_pb2 import EnvSpecsProto, MultiEnvSpecsProto
from shiva.core.communication_objects.metrics_pb2 import TrainingMetricsProto
from shiva.core.communication_objects.helpers_pb2 import SimpleMessage

@timed
def from_dict_2_ObservationsProto(observations: dict) -> ObservationsProto:
    assert "NotImplemented"
    observations_proto = ObservationsProto()

    return observations_proto

@timed
def from_ObservationsProto_2_dict(observations_proto: ObservationsProto) -> dict:
    assert "NotImplemented"
    observations = {}

    return observations

@timed
def from_dict_2_ActionsProto(actions: dict) -> ActionsProto:
    assert "NotImplemented"
    actions_proto = ActionsProto()

    return actions_proto

@timed
def from_ActionsProto_2_dict(actions_proto: ActionsProto) -> dict:
    assert "NotImplemented"
    actions = {}

    return actions

@timed
def from_dict_2_TrajectoriesProto(trajectory: dict) -> TrajectoriesProto:
    assert "NotImplemented"
    trajectories_proto = TrajectoriesProto()

    return trajectories_proto

@timed
def from_TrajectoriesProto_2_dict(trajectory: dict) -> dict:
    assert "NotImplemented"
    trajectories = {}

    return trajectories

@timed
def from_dict_2_NewAgentsConfigProto(agents: dict) -> ConfigProto:
    assert "NotImplemented"
    new_agents_config_proto = ConfigProto()

    return new_agents_config_proto

@timed
def from_NewAgentsConfigProto_2_dict(agents_proto: ConfigProto) -> dict:
    assert "NotImplemented"
    new_agents_config = {}

    return new_agents_config

@timed
def from_dict_2_TrainingMetricsProto(metrics: dict) -> TrainingMetricsProto:
    assert "NotImplemented"
    training_metrics_proto = TrainingMetricsProto()

    return training_metrics_proto

@timed
def from_EvolutionMetricProto_2_dict(metrics_proto: TrainingMetricsProto) -> dict:
    assert "NotImplemented"
    training_metrics = {}

    return training_metrics

@timed
def from_dict_2_EvolutionMetricProto(metrics: dict) -> TrainingMetricsProto:
    assert "NotImplemented"
    training_metrics_proto = TrainingMetricsProto()

    return training_metrics_proto

@timed
def from_TrainingMetricsProto_2_dict(metrics_proto: TrainingMetricsProto) -> dict:
    assert "NotImplemented"
    training_metrics = {}

    return training_metrics

@timed
def from_dict_2_EvolutionConfigProto(evol_config: dict) -> ConfigProto:
    assert "NotImplemented"
    evol_config_proto = ConfigProto()

    return evol_config_proto

@timed
def from_EvolutionConfigProto_2_dict(evol_config_proto: ConfigProto) -> dict:
    assert "NotImplemented"
    evol_config = {}

    return evol_config

@timed
def from_dict_2_MultiEnvSpecsProto(menv_specs: dict) -> StatusProto:
    assert "NotImplemented"
    menv_specs_proto = StatusProto()

    return menv_specs_proto

@timed
def from_MultiEnvSpecsProto_2_dict(menv_specs_proto_proto: StatusProto) -> dict:
    assert "NotImplemented"
    menv_specs = {}

    return menv_specs

@timed
def from_dict_2_EnvSpecsProto(env_specs: dict) -> StatusProto:
    assert "NotImplemented"
    env_specs_proto = StatusProto()

    return env_specs_proto

@timed
def from_EnvSpecsProto_to_dict(env_specs_proto: StatusProto) -> dict:
    assert "NotImplemented"
    env_specs = {}

    return env_specs

@timed
def from_dict_2_SimpleMessage(msg) -> SimpleMessage:
    assert "Implementation not checked"
    simple = SimpleMessage()
    simple.data = str(msg)
    return simple

@timed
def from_SimpleMessage_2_int(simple_msg_proto: SimpleMessage) -> int:
    assert "Implementation not checked"
    return int(simple_msg_proto.data)

@timed
def from_SimpleMessage_2_string(simple_msg_proto: SimpleMessage) -> str:
    assert "Implementation not checked"
    return str(simple_msg_proto.data)

# @timed
# def from_dict_2_LearnersInfoProto(learners_info: dict) -> LearnersInfoProto:
#     assert "NotImplemented"
#     learners_info_proto = LearnersInfoProto()
#
#     return learners_info_proto
#
# @timed
# def from_LearnersInfoProto_2_dict(learners_info_proto: LearnersInfoProto) -> dict:
#     assert "NotImplemented"
#     learners_info = {}
#
#     return learners_info

@timed
def from_dict_2_StatusProto(status_dict: dict) -> StatusProto:
    assert "NotImplemented"
    status_proto = StatusProto()

    return status_proto

@timed
def from_StatusProto_2_dict(status_proto: StatusProto) -> dict:
    assert "NotImplemented"
    status = {}

    return status




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