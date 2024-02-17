cmake_minimum_required(VERSION 3.0.2)
project(px4_mpc)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_python_setup()

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
   scripts/*
   nodes/*
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
