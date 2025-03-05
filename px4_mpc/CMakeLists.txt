cmake_minimum_required(VERSION 2.8.3)
project(px4_mpc)

find_package(catkin REQUIRED COMPONENTS
  rospy
  nav_msgs
  geometry_msgs
  visualization_msgs
  px4_msgs
  mpc_msgs
)

catkin_package()
