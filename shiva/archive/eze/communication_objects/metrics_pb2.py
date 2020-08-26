# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: shiva/core/communication_objects/metrics.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from shiva.core.communication_objects import enums_pb2 as shiva_dot_core_dot_communication__objects_dot_enums__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='shiva/core/communication_objects/metrics.proto',
  package='communication_objects',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n.shiva/core/communication_objects/metrics.proto\x12\x15\x63ommunication_objects\x1a,shiva/core/communication_objects/enums.proto\"\xbb\x01\n\x10\x41gentMetricProto\x12\x19\n\x11steps_per_episode\x18\x01 \x01(\x02\x12\x12\n\nstep_count\x18\x02 \x01(\x02\x12\x19\n\x11temp_done_counter\x18\x03 \x01(\x02\x12\x12\n\ndone_count\x18\x04 \x01(\x02\x12\x17\n\x0freward_per_step\x18\x05 \x01(\x02\x12\x1a\n\x12reward_per_episode\x18\x06 \x01(\x02\x12\x14\n\x0creward_total\x18\x07 \x01(\x02\"\xb1\x01\n\x14TrainingMetricsProto\x12\x43\n\x04\x64\x61ta\x18\x01 \x03(\x0b\x32\x35.communication_objects.TrainingMetricsProto.DataEntry\x1aT\n\tDataEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x36\n\x05value\x18\x02 \x01(\x0b\x32\'.communication_objects.AgentMetricProto:\x02\x38\x01\"\xb5\x01\n\x16\x45valuationMetricsProto\x12\x45\n\x04\x64\x61ta\x18\x01 \x03(\x0b\x32\x37.communication_objects.EvaluationMetricsProto.DataEntry\x1aT\n\tDataEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x36\n\x05value\x18\x02 \x01(\x0b\x32\'.communication_objects.AgentMetricProto:\x02\x38\x01\"\xe0\x02\n\x0cMetricsProto\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x32\n\x04type\x18\x02 \x01(\x0e\x32$.communication_objects.ComponentType\x12\x31\n\x06status\x18\x03 \x01(\x0e\x32!.communication_objects.StatusType\x12\x0f\n\x07\x61\x64\x64ress\x18\x04 \x01(\t\x12\x38\n\x05\x61gent\x18\x05 \x01(\x0b\x32\'.communication_objects.AgentMetricProtoH\x00\x12<\n\x05train\x18\x06 \x01(\x0b\x32+.communication_objects.TrainingMetricsProtoH\x00\x12=\n\x04\x65val\x18\x07 \x01(\x0b\x32-.communication_objects.EvaluationMetricsProtoH\x00\x12\r\n\x05\x65xtra\x18\x08 \x01(\tB\x06\n\x04\x64\x61tab\x06proto3')
  ,
  dependencies=[shiva_dot_core_dot_communication__objects_dot_enums__pb2.DESCRIPTOR,])




_AGENTMETRICPROTO = _descriptor.Descriptor(
  name='AgentMetricProto',
  full_name='communication_objects.AgentMetricProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='steps_per_episode', full_name='communication_objects.AgentMetricProto.steps_per_episode', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='step_count', full_name='communication_objects.AgentMetricProto.step_count', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='temp_done_counter', full_name='communication_objects.AgentMetricProto.temp_done_counter', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='done_count', full_name='communication_objects.AgentMetricProto.done_count', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reward_per_step', full_name='communication_objects.AgentMetricProto.reward_per_step', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reward_per_episode', full_name='communication_objects.AgentMetricProto.reward_per_episode', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reward_total', full_name='communication_objects.AgentMetricProto.reward_total', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=120,
  serialized_end=307,
)


_TRAININGMETRICSPROTO_DATAENTRY = _descriptor.Descriptor(
  name='DataEntry',
  full_name='communication_objects.TrainingMetricsProto.DataEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='communication_objects.TrainingMetricsProto.DataEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='communication_objects.TrainingMetricsProto.DataEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=403,
  serialized_end=487,
)

_TRAININGMETRICSPROTO = _descriptor.Descriptor(
  name='TrainingMetricsProto',
  full_name='communication_objects.TrainingMetricsProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='communication_objects.TrainingMetricsProto.data', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_TRAININGMETRICSPROTO_DATAENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=310,
  serialized_end=487,
)


_EVALUATIONMETRICSPROTO_DATAENTRY = _descriptor.Descriptor(
  name='DataEntry',
  full_name='communication_objects.EvaluationMetricsProto.DataEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='communication_objects.EvaluationMetricsProto.DataEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='communication_objects.EvaluationMetricsProto.DataEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=403,
  serialized_end=487,
)

_EVALUATIONMETRICSPROTO = _descriptor.Descriptor(
  name='EvaluationMetricsProto',
  full_name='communication_objects.EvaluationMetricsProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='communication_objects.EvaluationMetricsProto.data', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_EVALUATIONMETRICSPROTO_DATAENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=490,
  serialized_end=671,
)


_METRICSPROTO = _descriptor.Descriptor(
  name='MetricsProto',
  full_name='communication_objects.MetricsProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='communication_objects.MetricsProto.id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type', full_name='communication_objects.MetricsProto.type', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='status', full_name='communication_objects.MetricsProto.status', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='address', full_name='communication_objects.MetricsProto.address', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='agent', full_name='communication_objects.MetricsProto.agent', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='train', full_name='communication_objects.MetricsProto.train', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='eval', full_name='communication_objects.MetricsProto.eval', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='extra', full_name='communication_objects.MetricsProto.extra', index=7,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
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
    _descriptor.OneofDescriptor(
      name='data', full_name='communication_objects.MetricsProto.data',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=674,
  serialized_end=1026,
)

_TRAININGMETRICSPROTO_DATAENTRY.fields_by_name['value'].message_type = _AGENTMETRICPROTO
_TRAININGMETRICSPROTO_DATAENTRY.containing_type = _TRAININGMETRICSPROTO
_TRAININGMETRICSPROTO.fields_by_name['data'].message_type = _TRAININGMETRICSPROTO_DATAENTRY
_EVALUATIONMETRICSPROTO_DATAENTRY.fields_by_name['value'].message_type = _AGENTMETRICPROTO
_EVALUATIONMETRICSPROTO_DATAENTRY.containing_type = _EVALUATIONMETRICSPROTO
_EVALUATIONMETRICSPROTO.fields_by_name['data'].message_type = _EVALUATIONMETRICSPROTO_DATAENTRY
_METRICSPROTO.fields_by_name['type'].enum_type = shiva_dot_core_dot_communication__objects_dot_enums__pb2._COMPONENTTYPE
_METRICSPROTO.fields_by_name['status'].enum_type = shiva_dot_core_dot_communication__objects_dot_enums__pb2._STATUSTYPE
_METRICSPROTO.fields_by_name['agent'].message_type = _AGENTMETRICPROTO
_METRICSPROTO.fields_by_name['train'].message_type = _TRAININGMETRICSPROTO
_METRICSPROTO.fields_by_name['eval'].message_type = _EVALUATIONMETRICSPROTO
_METRICSPROTO.oneofs_by_name['data'].fields.append(
  _METRICSPROTO.fields_by_name['agent'])
_METRICSPROTO.fields_by_name['agent'].containing_oneof = _METRICSPROTO.oneofs_by_name['data']
_METRICSPROTO.oneofs_by_name['data'].fields.append(
  _METRICSPROTO.fields_by_name['train'])
_METRICSPROTO.fields_by_name['train'].containing_oneof = _METRICSPROTO.oneofs_by_name['data']
_METRICSPROTO.oneofs_by_name['data'].fields.append(
  _METRICSPROTO.fields_by_name['eval'])
_METRICSPROTO.fields_by_name['eval'].containing_oneof = _METRICSPROTO.oneofs_by_name['data']
DESCRIPTOR.message_types_by_name['AgentMetricProto'] = _AGENTMETRICPROTO
DESCRIPTOR.message_types_by_name['TrainingMetricsProto'] = _TRAININGMETRICSPROTO
DESCRIPTOR.message_types_by_name['EvaluationMetricsProto'] = _EVALUATIONMETRICSPROTO
DESCRIPTOR.message_types_by_name['MetricsProto'] = _METRICSPROTO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

AgentMetricProto = _reflection.GeneratedProtocolMessageType('AgentMetricProto', (_message.Message,), {
  'DESCRIPTOR' : _AGENTMETRICPROTO,
  '__module__' : 'shiva.core.communication_objects.metrics_pb2'
  # @@protoc_insertion_point(class_scope:communication_objects.AgentMetricProto)
  })
_sym_db.RegisterMessage(AgentMetricProto)

TrainingMetricsProto = _reflection.GeneratedProtocolMessageType('TrainingMetricsProto', (_message.Message,), {

  'DataEntry' : _reflection.GeneratedProtocolMessageType('DataEntry', (_message.Message,), {
    'DESCRIPTOR' : _TRAININGMETRICSPROTO_DATAENTRY,
    '__module__' : 'shiva.core.communication_objects.metrics_pb2'
    # @@protoc_insertion_point(class_scope:communication_objects.TrainingMetricsProto.DataEntry)
    })
  ,
  'DESCRIPTOR' : _TRAININGMETRICSPROTO,
  '__module__' : 'shiva.core.communication_objects.metrics_pb2'
  # @@protoc_insertion_point(class_scope:communication_objects.TrainingMetricsProto)
  })
_sym_db.RegisterMessage(TrainingMetricsProto)
_sym_db.RegisterMessage(TrainingMetricsProto.DataEntry)

EvaluationMetricsProto = _reflection.GeneratedProtocolMessageType('EvaluationMetricsProto', (_message.Message,), {

  'DataEntry' : _reflection.GeneratedProtocolMessageType('DataEntry', (_message.Message,), {
    'DESCRIPTOR' : _EVALUATIONMETRICSPROTO_DATAENTRY,
    '__module__' : 'shiva.core.communication_objects.metrics_pb2'
    # @@protoc_insertion_point(class_scope:communication_objects.EvaluationMetricsProto.DataEntry)
    })
  ,
  'DESCRIPTOR' : _EVALUATIONMETRICSPROTO,
  '__module__' : 'shiva.core.communication_objects.metrics_pb2'
  # @@protoc_insertion_point(class_scope:communication_objects.EvaluationMetricsProto)
  })
_sym_db.RegisterMessage(EvaluationMetricsProto)
_sym_db.RegisterMessage(EvaluationMetricsProto.DataEntry)

MetricsProto = _reflection.GeneratedProtocolMessageType('MetricsProto', (_message.Message,), {
  'DESCRIPTOR' : _METRICSPROTO,
  '__module__' : 'shiva.core.communication_objects.metrics_pb2'
  # @@protoc_insertion_point(class_scope:communication_objects.MetricsProto)
  })
_sym_db.RegisterMessage(MetricsProto)


_TRAININGMETRICSPROTO_DATAENTRY._options = None
_EVALUATIONMETRICSPROTO_DATAENTRY._options = None
# @@protoc_insertion_point(module_scope)
