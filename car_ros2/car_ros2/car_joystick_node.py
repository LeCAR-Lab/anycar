from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation
from tf_transformations import quaternion_from_euler, euler_matrix, euler_from_quaternion

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PolygonStamped, Point32
from car_ros2.utils import load_dynamic_params, load_mppi_params, load_env_params_mujoco, load_env_params_numeric, load_env_params_isaacsim, load_env_params_unity

class CarJoystickNode(Node):
    def __init__(self):
        super().__init__('car_joystick_node')
        self.model_params = load_dynamic_params()
        self.L = self.model_params.LF + self.model_params.LR
        self.vehicle_cmd_pub = self.create_publisher(AckermannDriveStamped, 'ackermann_command', 1)
        self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_callback, 1)
        self.joy = None
        self.timer = self.create_timer(0.02, self.timer_callback)
        self.ackermann_msg = None
        self.odom_sub = self.create_subscription(Odometry, 'odometry/filtered', self.odom_callback, 1)
        self.odom_copy_pub = self.create_publisher(Odometry, 'odometry_copy', 1)
        self.body_pub_ = self.create_publisher(PolygonStamped, 'body', 1)

    def joy_callback(self, msg):
        self.joy = msg

    def timer_callback(self):
        if self.joy is not None:
            self.ackermann_msg = AckermannDriveStamped()
            self.ackermann_msg.header.stamp = self.get_clock().now().to_msg()
            self.ackermann_msg.header.frame_id = 'base_link'
            self.ackermann_msg.drive.steering_angle = self.joy.axes[0]

            if self.joy.axes[2] <= 0.98:
                self.ackermann_msg.drive.speed = (self.joy.axes[2] - 1.0) / 2.0
            else:
                self.ackermann_msg.drive.speed = -1.0 * (self.joy.axes[5] - 1.0) / 2.0
            self.vehicle_cmd_pub.publish(self.ackermann_msg)

    def odom_callback(self, msg):
        if self.ackermann_msg is not None:
            # self.ackermann_msg.header.stamp = msg.header.stamp
            # self.vehicle_cmd_pub.publish(self.ackermann_msg)
            # self.odom_copy_pub.publish(msg)

            # body polygon
            pts = np.array([
                [self.model_params.LF, self.L/3],
                [self.model_params.LF, -self.L/3],
                [-self.model_params.LR, -self.L/3],
                [-self.model_params.LR, self.L/3],
            ])
            # transform to world frame
            psi = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
            px = msg.pose.pose.position.x
            py = msg.pose.pose.position.y
            R = euler_matrix(0, 0, psi)[:2, :2]
            pts = np.dot(R, pts.T).T
            pts += np.array([px, py])
            body = PolygonStamped()
            body.header.frame_id = 'map'
            body.header.stamp = msg.header.stamp
            for i in range(pts.shape[0]):
                p = Point32()
                p.x = float(pts[i, 0])
                p.y = float(pts[i, 1])
                p.z = 0.
                body.polygon.points.append(p)
            self.body_pub_.publish(body)


def main(args=None):
    rclpy.init(args=args)
    car_joystick_node = CarJoystickNode()
    rclpy.spin(car_joystick_node)
    car_joystick_node.destroy_node()
    rclpy.shutdown()