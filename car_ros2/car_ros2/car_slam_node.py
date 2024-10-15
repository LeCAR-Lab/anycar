from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation
from tf_transformations import quaternion_from_euler, euler_matrix, euler_from_quaternion
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point, PolygonStamped, Point32, Pose
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, Imu
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PolygonStamped, Point32
from car_ros2.utils import load_dynamic_params, load_mppi_params, load_env_params_mujoco, load_env_params_numeric, load_env_params_isaacsim, load_env_params_unity


SLAM_FREQUENCY = 50
class CarSlamNode(Node):
    def __init__(self):
        super().__init__('car_slam_node')
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 1)
        self.imu_sub = self.create_subscription(Imu, '/sensors/imu/raw', self.imu_callback, 1)
        self.pose_w_cov_sub = self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self.pose_w_cov_callback, 1)
        self.odom_pub_ = self.create_publisher(Odometry, 'odometry', 1)

        self.odom = None
        self.imu = None
        self.pose_w_cov = None
        
        self.timer = self.create_timer(1./SLAM_FREQUENCY, self.on_timer)
                
    def on_timer(self):
        # Store frame names in variables that will be used to
        # compute transformations
        if self.odom is not None and self.imu is not None and self.pose_w_cov is not None:
            odom_msg = deepcopy(self.odom)
            odom_msg.pose.pose.position.x = self.pose_w_cov.pose.pose.position.x
            odom_msg.pose.pose.position.y = self.pose_w_cov.pose.pose.position.y
            odom_msg.pose.pose.position.z = self.pose_w_cov.pose.pose.position.z
            odom_msg.pose.pose.orientation.x = self.pose_w_cov.pose.pose.orientation.x
            odom_msg.pose.pose.orientation.y = self.pose_w_cov.pose.pose.orientation.y
            odom_msg.pose.pose.orientation.z = self.pose_w_cov.pose.pose.orientation.z
            odom_msg.pose.pose.orientation.w = self.pose_w_cov.pose.pose.orientation.w
            odom_msg.twist.twist.angular.z = self.imu.angular_velocity.z
            self.odom_pub_.publish(odom_msg)
            
    def odom_callback(self, msg):
        self.odom = msg
        
    def imu_callback(self, msg):
        self.imu = msg
        

    def pose_w_cov_callback(self, msg):
        self.pose_w_cov = msg



def main(args=None):
    rclpy.init(args=args)
    car_slam_node = CarSlamNode()
    rclpy.spin(car_slam_node)
    car_slam_node.destroy_node()
    rclpy.shutdown()