cmake_minimum_required(VERSION 2.8.3)
project(georegistration_gazebo_msgs)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  trajectory_msgs
  geometry_msgs
  sensor_msgs
  std_srvs
  message_generation
  )

add_message_files(
  DIRECTORY msg
  FILES
  ModelSize.msg
  ModelSizeRequest.msg
  ModelSizeResponse.msg
  )

add_service_files(DIRECTORY srv FILES
  GetModelGeomProperties.srv
  GetModelSize.srv
  )

generate_messages(DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
  trajectory_msgs
  )

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  std_msgs
  trajectory_msgs
  geometry_msgs
  sensor_msgs
  std_srvs
  )
