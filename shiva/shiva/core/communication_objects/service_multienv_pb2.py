# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: shiva/core/communication_objects/service_multienv.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from shiva.core.communication_objects import configs_pb2 as shiva_dot_core_dot_communication__objects_dot_configs__pb2
from shiva.core.communication_objects import env_step_pb2 as shiva_dot_core_dot_communication__objects_dot_env__step__pb2
from shiva.core.communication_objects import helpers_pb2 as shiva_dot_core_dot_communication__objects_dot_helpers__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='shiva/core/communication_objects/service_multienv.proto',
  package='communication_objects',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n7shiva/core/communication_objects/service_multienv.proto\x12\x15\x63ommunication_objects\x1a.shiva/core/communication_objects/configs.proto\x1a/shiva/core/communication_objects/env_step.proto\x1a.shiva/core/communication_objects/helpers.proto2\xd1\x01\n\x10MultiEnvironment\x12\x61\n\x10SendObservations\x12(.communication_objects.ObservationsProto\x1a#.communication_objects.ActionsProto\x12Z\n\rSendNewAgents\x12+.communication_objects.NewAgentsConfigProto\x1a\x1c.communication_objects.Emptyb\x06proto3')
  ,
  dependencies=[shiva_dot_core_dot_communication__objects_dot_configs__pb2.DESCRIPTOR,shiva_dot_core_dot_communication__objects_dot_env__step__pb2.DESCRIPTOR,shiva_dot_core_dot_communication__objects_dot_helpers__pb2.DESCRIPTOR,])



_sym_db.RegisterFileDescriptor(DESCRIPTOR)



_MULTIENVIRONMENT = _descriptor.ServiceDescriptor(
  name='MultiEnvironment',
  full_name='communication_objects.MultiEnvironment',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=228,
  serialized_end=437,
  methods=[
  _descriptor.MethodDescriptor(
    name='SendObservations',
    full_name='communication_objects.MultiEnvironment.SendObservations',
    index=0,
    containing_service=None,
    input_type=shiva_dot_core_dot_communication__objects_dot_env__step__pb2._OBSERVATIONSPROTO,
    output_type=shiva_dot_core_dot_communication__objects_dot_env__step__pb2._ACTIONSPROTO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='SendNewAgents',
    full_name='communication_objects.MultiEnvironment.SendNewAgents',
    index=1,
    containing_service=None,
    input_type=shiva_dot_core_dot_communication__objects_dot_configs__pb2._NEWAGENTSCONFIGPROTO,
    output_type=shiva_dot_core_dot_communication__objects_dot_helpers__pb2._EMPTY,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_MULTIENVIRONMENT)

DESCRIPTOR.services_by_name['MultiEnvironment'] = _MULTIENVIRONMENT

# @@protoc_insertion_point(module_scope)
