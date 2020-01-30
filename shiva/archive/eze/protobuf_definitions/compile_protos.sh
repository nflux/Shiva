cd protobuf_definitions

SRC_DIR=protos/shiva/core/communication_objects
DST_DIR=../
PYTHON_PACKAGE=shiva/core/communication_objects

rm -rf $DST_DIR/$PYTHON_PACKAGE
mkdir -p $DST_DIR/$PYTHON_PACKAGE

# generate python message objects
python3 -m grpc_tools.protoc --proto_path=protos --python_out=$DST_DIR $SRC_DIR/*.proto
# generate python gRPC service object
GRPC=service_*
python3 -m grpc_tools.protoc --proto_path=protos --python_out=$DST_DIR --grpc_python_out=$DST_DIR $SRC_DIR/$GRPC
