cmake_minimum_required(VERSION 2.8.3)
project(px4_mpc)

find_package(catkin REQUIRED COMPONENTS
  rospy
  px4_msgs
  mpc_msgs
)

catkin_package()

catkin_install_python(PROGRAMS 
  scripts/mpc_quadrotor.py
  scripts/test_setpoints.py
  scripts/rviz_pos_marker.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
