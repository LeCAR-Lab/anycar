import socket
import pickle
import os
from car_foundation import CAR_FOUNDATION_MODEL_DIR
from car_foundation import CAR_FOUNDATION_DATA_DIR
from isaacsim_collect import ISAACSIM_COLLECT_TMP_DIR
from car_planner.track_generation import change_track
from car_dynamics.controllers_torch import AltPurePursuitController, RandWalkController
from car_planner.global_trajectory import GlobalTrajectory, generate_circle_trajectory, generate_oval_trajectory, generate_rectangle_trajectory, generate_raceline_trajectory
from car_dynamics.controllers_jax import MPPIController, rollout_fn_jax, MPPIRunningParams
from car_ros2.utils import load_mppi_params, load_dynamic_params
from car_dynamics.models_jax import DynamicsJax
from termcolor import colored
from scipy.spatial.transform import Rotation as R
import numpy as np
import jax
import jax.numpy as jnp
import datetime
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point, PolygonStamped, Point32, Pose
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, String
from visualization_msgs.msg import MarkerArray, Marker
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import Joy

import threading

from tf_transformations import quaternion_from_euler, euler_matrix, euler_from_quaternion
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class IsaacSimInterfaceNode(Node):
    def __init__(self):
        super().__init__('IsaacSim_Interface')

        self.parent_frame_id_ = (
            self.declare_parameter("odom_frame_id", "map")
            .get_parameter_value()
            .string_value
        )
        self.child_frame_id_ = (
            self.declare_parameter("odom_child_frame_id", "base_link")
            .get_parameter_value()
            .string_value
        )

        self.odom_pub_ = self.create_publisher(Odometry, 'odometry/filtered', 1)
        self.cmd_sub_ = self.create_subscription(AckermannDriveStamped, 'ackermann_command', self.cmd_callback, 1)
        self.tf_broadcaster_ = TransformBroadcaster(self)

        now = self.get_clock().now().to_msg()
        self.cmd = AckermannDriveStamped()
        self.cmd.header.stamp = now
        self.cmd.drive.speed = float(0)
        self.cmd.drive.steering_angle = float(0)

        self.controller_thread = threading.Thread(target=self.start_controller)
        self.controller_thread.start()

        self.cmd_count = 0

        # self.start_controller()
    
    def cmd_callback(self, msg:AckermannDriveStamped):
        self.cmd = msg
        self.cmd_count += 1

    def start_controller(self, host='localhost', port=65432):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            print(f"Connected to server at {host}:{port}")
            while rclpy.ok():
                data = s.recv(4096)
                [t, state]= pickle.loads(data)
                # print(f"Received data: {[t, state]}")
                
                pose_car = (state[0], state[1], 0)
                lin_vel_car = (state[3], state[4], 0)
                r = R.from_euler('xyz', (0, 0, state[2]))
                quat_car = r.as_quat() #xyzw
                # quat_car = np.array([quat_car[3], quat_car[0], quat_car[1], quat_car[2]])

                now = self.get_clock().now().to_msg()
                odom = Odometry()
                odom.header.frame_id = 'odom'
                odom.header.stamp = now
                odom.pose.pose.position.x = pose_car[0]
                odom.pose.pose.position.y = pose_car[1]
                odom.pose.pose.orientation.x = quat_car[0]
                odom.pose.pose.orientation.y = quat_car[1]
                odom.pose.pose.orientation.z = quat_car[2]
                odom.pose.pose.orientation.w = quat_car[3]
                odom.twist.twist.linear.x = lin_vel_car[0]
                odom.twist.twist.linear.y = lin_vel_car[1]
                odom.twist.twist.angular.z = state[5]
                self.odom_pub_.publish(odom)

                transform = TransformStamped()
                transform.header.stamp = now
                transform.header.frame_id = self.parent_frame_id_
                transform.child_frame_id = self.child_frame_id_
                transform.transform.translation.x = pose_car[0]
                transform.transform.translation.y = pose_car[1]
                transform.transform.translation.z = 0.0 #todo
                transform.transform.rotation.x = quat_car[0]
                transform.transform.rotation.y = quat_car[1]
                transform.transform.rotation.z = quat_car[2]
                transform.transform.rotation.w = quat_car[3]
                self.tf_broadcaster_.sendTransform(transform)

                # action = np.clip(action, env.action_space.low, env.action_space.high)
                if self.cmd_count < 3:
                    action = np.array([0., 0.])
                else:
                    steer = self.cmd.drive.steering_angle
                    throttle = self.cmd.drive.speed
                    action = [throttle, steer] 
                    # print("sent action", action)

                action = np.array(action, dtype=np.float32)
                s.sendall(pickle.dumps(action))

def main():
    rclpy.init()
    interface = IsaacSimInterfaceNode()
    rclpy.spin(interface)
    interface.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

