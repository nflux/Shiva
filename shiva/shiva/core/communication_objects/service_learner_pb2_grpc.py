# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from shiva.core.communication_objects import configs_pb2 as shiva_dot_core_dot_communication__objects_dot_configs__pb2
from shiva.core.communication_objects import env_step_pb2 as shiva_dot_core_dot_communication__objects_dot_env__step__pb2
from shiva.core.communication_objects import helpers_pb2 as shiva_dot_core_dot_communication__objects_dot_helpers__pb2
from shiva.core.communication_objects import specs_pb2 as shiva_dot_core_dot_communication__objects_dot_specs__pb2


class LearnerStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.SendMultiEnvSpecs = channel.unary_unary(
        '/communication_objects.Learner/SendMultiEnvSpecs',
        request_serializer=shiva_dot_core_dot_communication__objects_dot_specs__pb2.EnvSpecsProto.SerializeToString,
        response_deserializer=shiva_dot_core_dot_communication__objects_dot_helpers__pb2.Empty.FromString,
        )
    self.SendEvolutionConfig = channel.unary_unary(
        '/communication_objects.Learner/SendEvolutionConfig',
        request_serializer=shiva_dot_core_dot_communication__objects_dot_configs__pb2.ConfigProto.SerializeToString,
        response_deserializer=shiva_dot_core_dot_communication__objects_dot_helpers__pb2.Empty.FromString,
        )
    self.SendTrajectories = channel.unary_unary(
        '/communication_objects.Learner/SendTrajectories',
        request_serializer=shiva_dot_core_dot_communication__objects_dot_env__step__pb2.TrajectoriesProto.SerializeToString,
        response_deserializer=shiva_dot_core_dot_communication__objects_dot_helpers__pb2.Empty.FromString,
        )


class LearnerServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def SendMultiEnvSpecs(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def SendEvolutionConfig(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def SendTrajectories(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_LearnerServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'SendMultiEnvSpecs': grpc.unary_unary_rpc_method_handler(
          servicer.SendMultiEnvSpecs,
          request_deserializer=shiva_dot_core_dot_communication__objects_dot_specs__pb2.EnvSpecsProto.FromString,
          response_serializer=shiva_dot_core_dot_communication__objects_dot_helpers__pb2.Empty.SerializeToString,
      ),
      'SendEvolutionConfig': grpc.unary_unary_rpc_method_handler(
          servicer.SendEvolutionConfig,
          request_deserializer=shiva_dot_core_dot_communication__objects_dot_configs__pb2.ConfigProto.FromString,
          response_serializer=shiva_dot_core_dot_communication__objects_dot_helpers__pb2.Empty.SerializeToString,
      ),
      'SendTrajectories': grpc.unary_unary_rpc_method_handler(
          servicer.SendTrajectories,
          request_deserializer=shiva_dot_core_dot_communication__objects_dot_env__step__pb2.TrajectoriesProto.FromString,
          response_serializer=shiva_dot_core_dot_communication__objects_dot_helpers__pb2.Empty.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'communication_objects.Learner', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
