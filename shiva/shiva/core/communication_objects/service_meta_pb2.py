# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: shiva/core/communication_objects/service_meta.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from shiva.core.communication_objects import configs_pb2 as shiva_dot_core_dot_communication__objects_dot_configs__pb2
from shiva.core.communication_objects import specs_pb2 as shiva_dot_core_dot_communication__objects_dot_specs__pb2
from shiva.core.communication_objects import metrics_pb2 as shiva_dot_core_dot_communication__objects_dot_metrics__pb2
from shiva.core.communication_objects import helpers_pb2 as shiva_dot_core_dot_communication__objects_dot_helpers__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='shiva/core/communication_objects/service_meta.proto',
  package='communication_objects',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n3shiva/core/communication_objects/service_meta.proto\x12\x15\x63ommunication_objects\x1a.shiva/core/communication_objects/configs.proto\x1a,shiva/core/communication_objects/specs.proto\x1a.shiva/core/communication_objects/metrics.proto\x1a.shiva/core/communication_objects/helpers.proto2\x83\x03\n\x0bMetaLearner\x12N\n\nSendStatus\x12\".communication_objects.StatusProto\x1a\x1c.communication_objects.Empty\x12\\\n\x11SendMultiEnvSpecs\x12).communication_objects.MultiEnvSpecsProto\x1a\x1c.communication_objects.Empty\x12`\n\x13SendTrainingMetrics\x12+.communication_objects.TrainingMetricsProto\x1a\x1c.communication_objects.Empty\x12\x64\n\x15SendEvaluationMetrics\x12-.communication_objects.EvaluationMetricsProto\x1a\x1c.communication_objects.Emptyb\x06proto3')
  ,
  dependencies=[shiva_dot_core_dot_communication__objects_dot_configs__pb2.DESCRIPTOR,shiva_dot_core_dot_communication__objects_dot_specs__pb2.DESCRIPTOR,shiva_dot_core_dot_communication__objects_dot_metrics__pb2.DESCRIPTOR,shiva_dot_core_dot_communication__objects_dot_helpers__pb2.DESCRIPTOR,])



_sym_db.RegisterFileDescriptor(DESCRIPTOR)



_METALEARNER = _descriptor.ServiceDescriptor(
  name='MetaLearner',
  full_name='communication_objects.MetaLearner',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=269,
  serialized_end=656,
  methods=[
  _descriptor.MethodDescriptor(
    name='SendStatus',
    full_name='communication_objects.MetaLearner.SendStatus',
    index=0,
    containing_service=None,
    input_type=shiva_dot_core_dot_communication__objects_dot_configs__pb2._STATUSPROTO,
    output_type=shiva_dot_core_dot_communication__objects_dot_helpers__pb2._EMPTY,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='SendMultiEnvSpecs',
    full_name='communication_objects.MetaLearner.SendMultiEnvSpecs',
    index=1,
    containing_service=None,
    input_type=shiva_dot_core_dot_communication__objects_dot_specs__pb2._MULTIENVSPECSPROTO,
    output_type=shiva_dot_core_dot_communication__objects_dot_helpers__pb2._EMPTY,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='SendTrainingMetrics',
    full_name='communication_objects.MetaLearner.SendTrainingMetrics',
    index=2,
    containing_service=None,
    input_type=shiva_dot_core_dot_communication__objects_dot_metrics__pb2._TRAININGMETRICSPROTO,
    output_type=shiva_dot_core_dot_communication__objects_dot_helpers__pb2._EMPTY,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='SendEvaluationMetrics',
    full_name='communication_objects.MetaLearner.SendEvaluationMetrics',
    index=3,
    containing_service=None,
    input_type=shiva_dot_core_dot_communication__objects_dot_metrics__pb2._EVALUATIONMETRICSPROTO,
    output_type=shiva_dot_core_dot_communication__objects_dot_helpers__pb2._EMPTY,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_METALEARNER)

DESCRIPTOR.services_by_name['MetaLearner'] = _METALEARNER

# @@protoc_insertion_point(module_scope)
