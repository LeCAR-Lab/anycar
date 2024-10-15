from termcolor import colored
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point, PolygonStamped, Point32, Pose
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
# import tf_transformations

from car_ros2.utils import load_env_params_mujoco, load_env_params_numeric, load_env_params_isaacsim, load_env_params_unity
from car_dynamics.models_jax import DynamicBicycleModel
from car_dynamics.envs import make_env
import tf_transformations


import numpy as np
import math
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
print("DEVICE", jax.devices())

# CORRECT_SLAM = True
CORRECT_SLAM = False

class CarSimulatorNode(Node):
    def __init__(self):
        super().__init__("car_simulator_node")
        print("LAUNCHING SIMUALTION NODE")
        self.env_params = load_env_params_numeric()

        self.env = make_env(self.env_params)
        
        self.initialized = False

        
        if CORRECT_SLAM:
            self.odom_slam_pub = self.create_publisher(Odometry, "odom", 1)
            self.odom_slam_pub = self.create_publisher(PoseWithCovarianceStamped, "haoru_pose", 1)
        else:
            self.odom_pub = self.create_publisher(Odometry, "odometry", 1)
        self.timer_ = self.create_timer(5.0, self.timer_callback)
        self.vehicle_cmd_sub = self.create_subscription(
            AckermannDriveStamped, "ackermann_command", self.vehicle_cmd_callback, 1
        )
        self.tf_broadcaster = TransformBroadcaster(self)

    def timer_callback(self):
        if not self.initialized:
            obs = self.env.reset()
            px, py, psi, vx, vy, omega = self.env.obs_state().tolist()
            odom = Odometry()
            odom.header.stamp = self.get_clock().now().to_msg()
            odom.header.frame_id = "map"
            odom.child_frame_id = "base_link"
            odom.pose.pose.position.x = px
            odom.pose.pose.position.y = py
            odom.pose.pose.position.z = 0.0
            odom.pose.pose.orientation.x = 0.0
            odom.pose.pose.orientation.y = 0.0
            odom.pose.pose.orientation.z = math.sin(psi / 2)
            odom.pose.pose.orientation.w = math.cos(psi / 2)
            odom.twist.twist.linear.x = vx
            odom.twist.twist.linear.y = vy
            odom.twist.twist.angular.z = omega
            self.odom_pub.publish(odom)
        else:
            self.timer_.cancel()

    def vehicle_cmd_callback(self, msg):
        now = self.get_clock().now().to_msg()
        if not CORRECT_SLAM:
            self.initialized = True
            action = np.array([msg.drive.speed, msg.drive.steering_angle])
            obs, reward, done, info = self.env.step(action)
            px, py, psi, vx, vy, omega = self.env.obs_state().tolist()
            odom = Odometry()
            odom.header.stamp = now
            odom.header.frame_id = "map"
            odom.child_frame_id = "base_link"
            odom.pose.pose.position.x = px
            odom.pose.pose.position.y = py
            odom.pose.pose.position.z = 0.0
            odom.pose.pose.orientation.x = 0.0
            odom.pose.pose.orientation.y = 0.0
            odom.pose.pose.orientation.z = math.sin(psi / 2)
            odom.pose.pose.orientation.w = math.cos(psi / 2)
            odom.twist.twist.linear.x = vx
            odom.twist.twist.linear.y = vy
            odom.twist.twist.angular.z = omega
            self.odom_pub.publish(odom)

            transform = TransformStamped()
            transform.header.stamp = odom.header.stamp
            transform.header.frame_id = "map"
            transform.child_frame_id = "base_link"
            transform.transform.translation.x = px
            transform.transform.translation.y = py
            transform.transform.translation.z = 0.0
            transform.transform.rotation.x = 0.0
            transform.transform.rotation.y = 0.0
            transform.transform.rotation.z = math.sin(psi / 2)
            transform.transform.rotation.w = math.cos(psi / 2)
            self.tf_broadcaster.sendTransform(transform)
        else:
        
            self.initialized = True
            action = np.array([msg.drive.speed, msg.drive.steering_angle])
            obs, reward, done, info = self.env.step(action)
            px, py, psi, vx, vy, omega = self.env.obs_state().tolist()
            
            odom_ground_truth = Odometry()
            odom_ground_truth.header.stamp = now
            odom_ground_truth.header.frame_id = "map"
            odom_ground_truth.child_frame_id = "base_link"
            odom_ground_truth.pose.pose.position.x = px
            odom_ground_truth.pose.pose.position.y = py
            odom_ground_truth.pose.pose.position.z = 0.0
            odom_ground_truth.pose.pose.orientation.x = 0.0
            odom_ground_truth.pose.pose.orientation.y = 0.0
            odom_ground_truth.pose.pose.orientation.z = math.sin(psi / 2)
            odom_ground_truth.pose.pose.orientation.w = math.cos(psi / 2)
            odom_ground_truth.twist.twist.linear.x = vx
            odom_ground_truth.twist.twist.linear.y = vy
            odom_ground_truth.twist.twist.angular.z = omega
            self.odom_slam_pub.publish(odom_ground_truth)       
            
            haoru_slam = PoseWithCovarianceStamped()
            haoru_slam.header.stamp = now
            haoru_slam.header.frame_id = "map"
            haoru_slam.pose.pose.position.x = px
            haoru_slam.pose.pose.position.y = py
            haoru_slam.pose.pose.position.z = 0.0
            haoru_slam.pose.pose.orientation.x = 0.0
            haoru_slam.pose.pose.orientation.y = 0.0
            haoru_slam.pose.pose.orientation.z = math.sin(psi / 2)
            haoru_slam.pose.pose.orientation.w = math.cos(psi / 2)
            self.odom_slam_pub.publish(haoru_slam)
            


def main(args=None):
    # sim_type = 'numerical'
    rclpy.init(args=args)

    car_simulator_node = CarSimulatorNode()

    rclpy.spin(car_simulator_node)

    car_simulator_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
