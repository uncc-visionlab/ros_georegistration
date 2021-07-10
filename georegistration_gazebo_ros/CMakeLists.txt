cmake_minimum_required(VERSION 2.8.3)
project(georegistration_gazebo_ros)

set(CMAKE_VERBOSE_MAKEFILE ON)
find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  roscpp
  geometry_msgs
  std_srvs
  tf
  std_msgs
  gazebo_msgs
  georegistration_gazebo_msgs)

find_package(Boost COMPONENTS chrono context coroutine date_time filesystem
               iostreams regex serialization system thread REQUIRED)
find_package(gazebo REQUIRED)

catkin_package(
  LIBRARIES
    gazebo_ros_api_plugin2
  CATKIN_DEPENDS
    roscpp
    geometry_msgs
    std_srvs
    tf
    std_msgs
    gazebo_msgs
    georegistration_gazebo_msgs
  )

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS})

link_directories(${catkin_LIBRARY_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
#set(cxx_flags)
#foreach (item ${GAZEBO_CFLAGS})
#  set(cxx_flags "${cxx_flags} ${item}")
#endforeach ()

#set(ld_flags)
#foreach (item ${GAZEBO_LDFLAGS})
#  set(ld_flags "${ld_flags} ${item}")
#endforeach ()
#message("Includes: ${CXX_INCLUDES}")

## Plugins
add_library(georegistration_gazebo_ros_api_plugin src/georegistration_gazebo_ros_api_plugin.cpp)
#add_dependencies(gazebo_ros_api_plugin2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(georegistration_gazebo_ros_api_plugin ${catkin_EXPORTED_TARGETS})
#set_target_properties(georegistration_gazebo_ros_api_plugin PROPERTIES LINK_FLAGS "${ld_flags}")
#set_target_properties(gazebo_ros_api_plugin2 PROPERTIES COMPILE_FLAGS "${cxx_flags}")
target_link_libraries(georegistration_gazebo_ros_api_plugin ${catkin_LIBRARIES} ${Boost_LIBRARIES})
