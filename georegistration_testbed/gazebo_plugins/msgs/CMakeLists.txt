find_package(Protobuf REQUIRED)

set(PROTOBUF_IMPORT_DIR)
foreach(ITR ${GAZEBO_INCLUDE_DIRS})
  if(ITR MATCHES ".*gazebo-[0-9.]+$")
    set(PROTOBUF_IMPORT_DIR "${ITR}/gazebo/msgs/proto")
  endif()
endforeach()

set (new_msgs
  model_bbox_request.proto
  model_bbox_response.proto
)

set (msgs
  model_bbox_request.proto
  model_bbox_response.proto
  ${PROTOBUF_IMPORT_DIR}/vector2d.proto
  ${PROTOBUF_IMPORT_DIR}/vector3d.proto
  #${PROTOBUF_IMPORT_DIR}/header.proto
  #${PROTOBUF_IMPORT_DIR}/time.proto
)

#include (
#    ${GAZEBO_INCLUDE_DIR}
#)

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${msgs})
add_library(model_bbox_msgs SHARED ${PROTO_SRCS})
target_link_libraries(model_bbox_msgs ${PROTOBUF_LIBRARY})

