
SRC_DIR=shiva/core/protobuf_definitions/protos
DST_DIR=shiva/core/communication_objects

rm -rf $DST_DIR
mkdir -p $DST_DIR

python3 -m grpc_tools.protoc -I=$SRC_DIR --python_out=$DST_DIR --grpc_python_out=$DST_DIR $SRC_DIR/all.proto