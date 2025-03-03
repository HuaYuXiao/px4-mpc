cmake_minimum_required(VERSION 2.8.3)
project(mpc_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  geometry_msgs
)

add_service_files(FILES
  SetPose.srv
)

generate_messages(
  DEPENDENCIES geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime geometry_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
