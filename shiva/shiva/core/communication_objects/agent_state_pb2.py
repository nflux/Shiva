# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: shiva/core/communication_objects/agent_state.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from shiva.core.communication_objects import helpers_pb2 as shiva_dot_core_dot_communication__objects_dot_helpers__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='shiva/core/communication_objects/agent_state.proto',
  package='communication_objects',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n2shiva/core/communication_objects/agent_state.proto\x12\x15\x63ommunication_objects\x1a.shiva/core/communication_objects/helpers.proto\"u\n\nAgentState\x12\n\n\x02id\x18\x01 \x01(\x05\x12=\n\x10next_observation\x18\x02 \x01(\x0b\x32#.communication_objects.ListOfFloats\x12\x0e\n\x06reward\x18\x03 \x01(\x02\x12\x0c\n\x04\x64one\x18\x04 \x01(\x08\x62\x06proto3')
  ,
  dependencies=[shiva_dot_core_dot_communication__objects_dot_helpers__pb2.DESCRIPTOR,])




_AGENTSTATE = _descriptor.Descriptor(
  name='AgentState',
  full_name='communication_objects.AgentState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='communication_objects.AgentState.id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='next_observation', full_name='communication_objects.AgentState.next_observation', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reward', full_name='communication_objects.AgentState.reward', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='done', full_name='communication_objects.AgentState.done', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=125,
  serialized_end=242,
)

_AGENTSTATE.fields_by_name['next_observation'].message_type = shiva_dot_core_dot_communication__objects_dot_helpers__pb2._LISTOFFLOATS
DESCRIPTOR.message_types_by_name['AgentState'] = _AGENTSTATE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

AgentState = _reflection.GeneratedProtocolMessageType('AgentState', (_message.Message,), {
  'DESCRIPTOR' : _AGENTSTATE,
  '__module__' : 'shiva.core.communication_objects.agent_state_pb2'
  # @@protoc_insertion_point(class_scope:communication_objects.AgentState)
  })
_sym_db.RegisterMessage(AgentState)


# @@protoc_insertion_point(module_scope)
