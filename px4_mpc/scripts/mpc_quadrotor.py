#!/usr/bin/python3
############################################################################
#
#   Copyright (C) 2023 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

from models.multirotor_rate_model import MultirotorRateModel
from controllers.multirotor_rate_mpc import MultirotorRateMPC

__author__ = "Jaeyoung Lim"
__contact__ = "jalim@ethz.ch"

import rospy
import numpy as np

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker

from px4_msgs.msg import OffboardControlMode, VehicleStatus, VehicleAttitude, VehicleLocalPosition, VehicleRatesSetpoint
from mpc_msgs.srv import SetPose, SetPoseResponse

def vector2PoseMsg(frame_id, position, attitude):
    pose_msg = PoseStamped()
    pose_msg.header.stamp = rospy.Time.now()
    pose_msg.header.frame_id = frame_id
    pose_msg.pose.orientation.w = attitude[0]
    pose_msg.pose.orientation.x = attitude[1]
    pose_msg.pose.orientation.y = attitude[2]
    pose_msg.pose.orientation.z = attitude[3]
    pose_msg.pose.position.x = float(position[0])
    pose_msg.pose.position.y = float(position[1])
    pose_msg.pose.position.z = float(position[2])
    return pose_msg

class QuadrotorMPC(object):
    def __init__(self):
        # Subscribers
        self.status_sub = rospy.Subscriber('/fmu/out/vehicle_status', VehicleStatus, self.vehicle_status_callback)
        self.attitude_sub = rospy.Subscriber('/fmu/out/vehicle_attitude', VehicleAttitude, self.vehicle_attitude_callback)
        self.local_position_sub = rospy.Subscriber('/fmu/out/vehicle_local_position', VehicleLocalPosition, self.vehicle_local_position_callback)
        
        # Service
        self.set_pose_srv = rospy.Service('/set_pose', SetPose, self.add_set_pos_callback)
        
        # Publishers
        self.publisher_offboard_mode = rospy.Publisher('/fmu/in/offboard_control_mode', OffboardControlMode, queue_size=1)
        self.publisher_rates_setpoint = rospy.Publisher('/fmu/in/vehicle_rates_setpoint', VehicleRatesSetpoint, queue_size=1)
        self.predicted_path_pub = rospy.Publisher('/px4_mpc/predicted_path', Path, queue_size=10)
        self.reference_pub = rospy.Publisher('/px4_mpc/reference', Marker, queue_size=10)
        
        # Timer (ROS1 uses rospy.Timer)
        self.timer = rospy.Timer(rospy.Duration(0.02), self.cmdloop_callback)
        
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        
        # Instantiate model and controller
        self.model = MultirotorRateModel()
        MPC_HORIZON = 15
        self.mpc = MultirotorRateMPC(self.model)
        
        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        self.setpoint_position = np.array([0.0, 0.0, 3.0])
    
    def vehicle_attitude_callback(self, msg):
        # TODO: handle NED->ENU transformation if needed
        self.vehicle_attitude[0] = msg.q[0]
        self.vehicle_attitude[1] = msg.q[1]
        self.vehicle_attitude[2] = -msg.q[2]
        self.vehicle_attitude[3] = -msg.q[3]
    
    def vehicle_local_position_callback(self, msg):
        # TODO: handle NED->ENU transformation if needed
        self.vehicle_local_position[0] = msg.x
        self.vehicle_local_position[1] = -msg.y
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vx
        self.vehicle_local_velocity[1] = -msg.vy
        self.vehicle_local_velocity[2] = -msg.vz
    
    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state
    
    def publish_reference(self, pub, reference):
        msg = Marker()
        msg.action = Marker.ADD
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.ns = "arrow"
        msg.id = 1
        msg.type = Marker.SPHERE
        msg.scale.x = 0.5
        msg.scale.y = 0.5
        msg.scale.z = 0.5
        msg.color.r = 1.0
        msg.color.g = 0.0
        msg.color.b = 0.0
        msg.color.a = 1.0
        msg.pose.position.x = reference[0]
        msg.pose.position.y = reference[1]
        msg.pose.position.z = reference[2]
        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        pub.publish(msg)
    
    def cmdloop_callback(self, event):
        # Publish offboard control mode
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(rospy.Time.now().to_nsec() / 1000)
        offboard_msg.position = False
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = True
        self.publisher_offboard_mode.publish(offboard_msg)
        
        error_position = self.vehicle_local_position - self.setpoint_position
        x0 = np.array([
            error_position[0], error_position[1], error_position[2],
            self.vehicle_local_velocity[0], self.vehicle_local_velocity[1], self.vehicle_local_velocity[2],
            self.vehicle_attitude[0], self.vehicle_attitude[1], self.vehicle_attitude[2], self.vehicle_attitude[3]
        ]).reshape(10, 1)
        
        u_pred, x_pred = self.mpc.solve(x0)
        
        predicted_path_msg = Path()
        predicted_path_msg.header.stamp = rospy.Time.now()
        predicted_path_msg.header.frame_id = "map"
        for predicted_state in x_pred:
            predicted_pose_msg = vector2PoseMsg('map', predicted_state[0:3] + self.setpoint_position, np.array([1.0, 0.0, 0.0, 0.0]))
            predicted_path_msg.poses.append(predicted_pose_msg)
        self.predicted_path_pub.publish(predicted_path_msg)
        self.publish_reference(self.reference_pub, self.setpoint_position)
        
        thrust_rates = u_pred[0, :]
        # Hover thrust = 0.73 (example)
        thrust_command = -(thrust_rates[0] * 0.07 + 0.0)
        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            setpoint_msg = VehicleRatesSetpoint()
            setpoint_msg.timestamp = int(rospy.Time.now().to_nsec() / 1000)
            setpoint_msg.roll = float(thrust_rates[1])
            setpoint_msg.pitch = float(-thrust_rates[2])
            setpoint_msg.yaw = float(-thrust_rates[3])
            # Ensure thrust_body is properly initialized (assuming a list of 3 floats)
            setpoint_msg.thrust_body = [0.0, 0.0, float(thrust_command)]
            self.publisher_rates_setpoint.publish(setpoint_msg)
    
    def add_set_pos_callback(self, req):
        self.setpoint_position[0] = req.pose.position.x
        self.setpoint_position[1] = req.pose.position.y
        self.setpoint_position[2] = req.pose.position.z
        return SetPoseResponse()

def main():
    rospy.init_node('minimal_publisher', anonymous=True)
    quadrotor_mpc = QuadrotorMPC()
    rospy.spin()

if __name__ == '__main__':
    main()
