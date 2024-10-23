from copy import deepcopy
import torch
from termcolor import colored
import rclpy
from rclpy.node import Node
import time

from car_ros2 import CAR_ROS2_TMP
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point, PolygonStamped, Point32, Pose
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, String
from visualization_msgs.msg import MarkerArray, Marker
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import Joy

from tf_transformations import quaternion_from_euler, euler_matrix, euler_from_quaternion

from car_planner import CAR_PLANNER_ASSETS_DIR
from car_dynamics.models_jax import DynamicBicycleModel
from car_dynamics.controllers_jax import MPPIController, rollout_fn_select, MPPIRunningParams, void_fn
from car_dynamics.controllers_torch import PurePersuitParams, PurePersuitController
from car_planner.global_trajectory import GlobalTrajectory, generate_circle_trajectory, generate_oval_trajectory, generate_rectangle_trajectory, generate_raceline_trajectory
from car_dynamics.models_jax import DynamicsJax
from car_foundation import CAR_FOUNDATION_MODEL_DIR
import numpy as np
import jax
import jax.numpy as jnp
import tf2_geometry_msgs
import datetime


print("DEVICE", jax.devices())

from car_ros2.utils import load_dynamic_params, load_mppi_params, load_env_params_mujoco, load_env_params_numeric, load_env_params_isaacsim, load_env_params_unity

import threading
from multiprocessing.pool import ThreadPool


from car_dynamics.models_jax.dbm import CarState, CarAction

unique_prefix = datetime.datetime.now().isoformat(timespec='milliseconds')

SPEED = 1.
SAFE_SPEED_MAX = 10.0

# TELEOP = True
TELEOP = False
USE_KEYBOARD = False


if USE_KEYBOARD:
    from pynput import keyboard

import os
os.environ["OMP_NUM_THREADS"] = "1"


class CarNode(Node):
    def __init__(self):
        super().__init__('car_node')
        
        # print("Car node start")
        self.env_params = load_env_params_numeric()
        self.model_params = load_dynamic_params()
        self.mppi_params = load_mppi_params()

        # NOTE: Can choose either 'mppi' or 'pure_persuit'
        # self.controller_type = 'pure_persuit'
        self.controller_type = 'mppi'
        
        self.L = self.model_params.LF + self.model_params.LR
        self._counter = 0        
        print("DYANMICS", self.mppi_params.dynamics)
        if self.controller_type == 'mppi':
            ## MPPI Configself.mppi_running_params_warmup
            DYNAMICS = self.mppi_params.dynamics
            if DYNAMICS == "dbm":
                self.dynamics = DynamicBicycleModel(self.model_params)
                self.dynamics.reset()
                self.rollout_fn = rollout_fn_select('dbm', self.dynamics, self.model_params.DT, self.L, self.model_params.LR)
            elif DYNAMICS == "transformer-torch":
                from car_dynamics.models_torch.nn_dynamics import DynamicsTorch
                ## Load Transformer
                self.dynamics = DynamicsTorch({DynamicsJax({'model_path':os.path.join(CAR_FOUNDATION_MODEL_DIR, "2024-07-15T17:56:55.014-model_checkpoint", f"{400}", "default")})})
                # self.dynamics = DynamicsJax({})
                print(colored(type(self.dynamics), "blue"))
                self.rollout_fn = rollout_fn_select('transformer-torch', self.dynamics, self.model_params.DT, self.L, self.model_params.LR)
            elif DYNAMICS == "transformer-jax":
                ## Load Transformer
                self.dynamics = DynamicsJax({
                    "model_path": os.path.join(CAR_FOUNDATION_MODEL_DIR, "anycar_model_checkpoint/500/default"), # pt
                })
                print(colored("Loaded JAX transformer model", "green"))
                print(colored(type(self.dynamics), "blue"))
                self.rollout_fn = rollout_fn_select('transformer-jax', self.dynamics, self.model_params.DT, self.L, self.model_params.LR)
            else:
                raise ValueError(f"Invalid dynamics model: {DYNAMICS}")
            
            
            self.key = jax.random.PRNGKey(0)
            key, self.key = jax.random.split(self.key)
            self.mppi = MPPIController(
                self.mppi_params, self.rollout_fn, void_fn, key
            )
            
            self.mppi_running_params = self.mppi.get_init_params()
            
            self.key, key2 = jax.random.split(self.key)
            
            
            self.mppi_running_params = MPPIRunningParams(
                a_mean = self.mppi_running_params.a_mean,
                a_cov = self.mppi_running_params.a_cov,
                prev_a = self.mppi_running_params.prev_a,
                state_hist = self.mppi_running_params.state_hist,
                key = key2,
            )
            
            ## Define the warmup MPPI Based on DBM model
            self.mppi_params_warmup = load_mppi_params()
            self.mppi_params_warmup.dynamics = 'dbm'
            self.dynamics_warmup = DynamicBicycleModel(self.model_params)
            self.dynamics_warmup.reset()
            self.rollout_fn_warmup = rollout_fn_select('dbm', self.dynamics_warmup, self.model_params.DT, self.L, self.model_params.LR)
            self.key, key2 = jax.random.split(self.key, 2)
            self.mppi_warmup = MPPIController(self.mppi_params_warmup, self.rollout_fn_warmup, void_fn, key2)
            self.mppi_running_params_warmup = self.mppi_warmup.get_init_params()
            self.key, key2 = jax.random.split(self.key, 2)
            self.mppi_running_params_warmup = MPPIRunningParams(
                a_mean = self.mppi_running_params_warmup.a_mean,
                a_cov = self.mppi_running_params_warmup.a_cov,
                prev_a = self.mppi_running_params_warmup.prev_a,
                state_hist = self.mppi_running_params_warmup.state_hist,
                key = key2,
            )         
        elif self.controller_type == 'pure_persuit':        
            ## Pure pursuit controller
            pure_persuit_params = PurePersuitParams()
            if 'numeric' in self.env_params.name or \
                'mujoco' in self.env_params.name or \
                    'unity' in self.env_params.name or \
                        'isaac' in self.env_params.name:
                pure_persuit_params.mode = 'throttle'
                pure_persuit_params.target_vel = 0.8
                pure_persuit_params.wheelbase = 0.2
                pure_persuit_params.kp = 3.
            self.pure_pursuit = PurePersuitController(pure_persuit_params)
            
        pure_persuit_params = PurePersuitParams()
        if 'numeric' in self.env_params.name or \
                'mujoco' in self.env_params.name or\
                    'unity' in self.env_params.name or \
                        'isaac' in self.env_params.name:
            pure_persuit_params.mode = 'throttle'
            pure_persuit_params.target_vel = 0.8
            pure_persuit_params.wheelbase = 0.2
            pure_persuit_params.kp = 3.
        self.recover_controller = PurePersuitController(pure_persuit_params)

        # Load pre-computed track file
        # Here are three examples of tracks
        # 1. .txt
        # 2. gnerate_fn
        # 3. .csv
        # track = np.loadtxt(os.path.join(CAR_PLANNER_ASSETS_DIR, "math_park_v2.txt"), delimiter=',', skiprows=1)
        # track = generate_oval_trajectory((0., -20.0), 20.0, 20.0, direction=-1)
        track = np.loadtxt(os.path.join(CAR_PLANNER_ASSETS_DIR, "cuc_inside.csv"), delimiter=',', skiprows=1)
        
        self.global_planner = GlobalTrajectory(track)
        self.step_mode_ = self.declare_parameter('step_mode', False).value
        ## ROS2 publishers and subscribers
        self.path_pub_ = self.create_publisher(Path, 'path', 1)
        self.waypoint_list_pub_ = self.create_publisher(Path, 'waypoint_list', 1)
        self.ref_trajectory_pub_ = self.create_publisher(Path, 'ref_trajectory', 1)
        self.pose_pub_ = self.create_publisher(PoseWithCovarianceStamped, 'pose', 1)
        # if not self.step_mode_:
        #     self.timer_ = self.create_timer(self.model_params.DT, self.timer_callback)
        self.slow_timer_ = self.create_timer(1.0, self.slow_timer_callback)
        self.throttle_pub_ = self.create_publisher(Float64, 'speed', 1)
        self.steer_pub_ = self.create_publisher(Float64, 'steering', 1)
        self.trajectory_array_pub_ = self.create_publisher(MarkerArray, 'trajectory_array', 1)
        self.body_pub_ = self.create_publisher(PolygonStamped, 'body', 1)
        self.vehicle_cmd_pub_ = self.create_publisher(AckermannDriveStamped, 'ackermann_command', 1)
        self.odom_sub_ = self.create_subscription(Odometry, 'odometry', self.odom_callback, 1)
        self.odom_copy_pub_ = self.create_publisher(Odometry, 'odometry_copy', 1)
        self.action_rate_pub = self.create_publisher(Float64, 'debug/action_rate', 1)
        self.lateral_error_pub_ = self.create_publisher(Float64, 'lateral_error', 1)
        self.ref_vel_pub_ = self.create_publisher(Odometry, 'tracking/ref_vel', 1)
        self.loss_pub_ = self.create_publisher(Float64, 'adapt/loss', 1)
        self.misc_pub_ = self.create_publisher(String, 'misc_message', 1)
        self.mppi_time_pub_ = self.create_publisher(Float64, 'mppi_time', 1)
        if TELEOP:
            self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_callback, 1)
            self.joy = None
        self.odom = None
        self.prev_action = np.zeros(2)
        self.debug_buffer = dict(timestamp=[], obs=[], action=[], action_canidate=[], sampled_traj=[])
        
        self.params_pub_list = []
        for param in self.model_params.to_dict().keys():
            self.params_pub_list.append(self.create_publisher(Float64, f"param/{param}", 1))


        # self.is_recover_mode = False
        self.is_recover_mode = False
        self.emergency_stop = False
        if USE_KEYBOARD:
            listener = keyboard.Listener(on_press=self.on_press_key)
            listener.start()


        if not self.step_mode_:
            timer_thread = threading.Thread(target=self.timer_thread_fn)
            timer_thread.start()
        
        
    def on_press_key(self, key):
        # print(key, type(key), dir(key), key.char)
        if hasattr(key, 'char') and key.char == 'r':
            self.is_recover_mode = not self.is_recover_mode
            print(colored(f"[INFO] Recover mode: {self.is_recover_mode}", "blue"))
        if hasattr(key, 'char') and key.char == 'q':
            self.emergency_stop = not self.emergency_stop
            if self.emergency_stop:
                print(colored(f"[INFO] Emergency stop", "red"))
        
        
    def timer_thread_fn(self):
        while True:
            self.timer_callback()
        
    def timer_callback(self):
        
        start_time = self.get_clock().now()
        # print("here")
        if self.odom is None:
            print("ODOM NOT FOUND!")
            # time.sleep(self.model_params.DT)
            return
        
        if TELEOP and self.joy is None:
            print("TELEOP NOT FOUND!")
            return
    
        
        odom_copy = deepcopy(self.odom)

        rpy = euler_from_quaternion([
            self.odom.pose.pose.orientation.x,
            self.odom.pose.pose.orientation.y,
            self.odom.pose.pose.orientation.z,
            self.odom.pose.pose.orientation.w,
        ])
        
        pose_car = np.array([
            self.odom.pose.pose.position.x,
            self.odom.pose.pose.position.y,
            self.odom.pose.pose.position.z,
        ], dtype=np.float32)
        
        lin_vel_car = np.array([
            self.odom.twist.twist.linear.x,
            self.odom.twist.twist.linear.y,
            self.odom.twist.twist.linear.z,
        ], dtype=np.float32)
        
        quat_car = np.array([
            self.odom.pose.pose.orientation.w,
            self.odom.pose.pose.orientation.x,
            self.odom.pose.pose.orientation.y,
            self.odom.pose.pose.orientation.z,
        ], dtype=np.float32)
        
        state = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, rpy[2], self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.angular.z], dtype=np.float32)
        
        ## Initialize the history
        if self._counter == 0:        # state = env.obs_state()
            if self.controller_type == 'mppi':
                for _ in range(250):
                    self.mppi_running_params = self.mppi.feed_hist(self.mppi_running_params, state, np.array([0., 0.]))
        
        if np.any(np.isnan(state)):
            return
        
        # print("State", state)
        ## Generate reference trajectory
        target_pos_arr, frenet_pose = self.global_planner.generate(state[:5], self.model_params.DT, (self.mppi_params.h_knot - 1) * self.mppi_params.num_intermediate + 2 + self.mppi_params.delay, True)
        target_pos_arr[:, 3] = np.clip(target_pos_arr[:, 3], 0.0, SAFE_SPEED_MAX)
        target_pos_list = np.array(target_pos_arr)


        ref_vel = target_pos_arr[0][3]
        action_candidate_np = None
        sampled_traj = None
        if self.controller_type == 'mppi':
            if hasattr(self.dynamics, "reset"):
                self.dynamics.reset()
            _ctr_start = time.time()    
            target_pos_tensor = jnp.array(target_pos_arr)
            dynamic_params_tuple = (self.model_params.LF, self.model_params.LR, self.model_params.MASS, self.model_params.DT, self.model_params.K_RFY, self.model_params.K_FFY, self.model_params.Iz, self.model_params.Ta, self.model_params.Tb, self.model_params.Sa, self.model_params.Sb, self.model_params.mu, self.model_params.Cf, self.model_params.Cr, self.model_params.Bf, self.model_params.Br, self.model_params.hcom, self.model_params.fr)
            
            if self.mppi_params.dual and self._counter % 1 == 0:
                # DUAL MPPI AS WARMUP
                self.mppi_running_params_warmup = MPPIRunningParams(
                    a_mean = self.mppi_running_params.a_mean,
                    a_cov = self.mppi_running_params_warmup.a_cov,
                    prev_a = self.mppi_running_params_warmup.prev_a,
                    state_hist = self.mppi_running_params_warmup.state_hist,
                    key = self.mppi_running_params_warmup.key,
                )
                
                _, self.mppi_running_params_warmup, _ = self.mppi_warmup(state, target_pos_tensor, self.mppi_running_params_warmup, dynamic_params_tuple)
            
                self.mppi_running_params = MPPIRunningParams(
                    a_mean = (self.mppi_running_params_warmup.a_mean + self.mppi_running_params.a_mean) / 2,
                    a_cov = self.mppi_running_params.a_cov,
                    prev_a = self.mppi_running_params.prev_a,
                    state_hist = self.mppi_running_params.state_hist,
                    key = self.mppi_running_params.key,
                )
            
            action, self.mppi_running_params, mppi_info = self.mppi(state,target_pos_tensor,self.mppi_running_params, dynamic_params_tuple)

            st_ = time.time()
            action = np.array(action, dtype=np.float32)
            
            action_candidate_np = np.array(mppi_info['a_mean_jnp'])
            sampled_traj = np.array(mppi_info['trajectory'][:, :2])  
            print("ctr time", time.time() - _ctr_start)

        elif self.controller_type == 'pure_persuit':
            # print(colored("Pure Persuit Controller", "green"))
            action  = self.pure_pursuit.step(pose_car, lin_vel_car, quat_car, target_pos_list)
        else:
            raise ValueError(f"Invalid controller type: {self.controller_type}")        
        
        if self.is_recover_mode: # Override the action with recover controller
            action = self.recover_controller.step(pose_car, lin_vel_car, quat_car, target_pos_list)
        
        if TELEOP:
            # Map joystick to action
            steer_joy = self.joy.axes[0]

            if self.joy.axes[2] <= 0.98:
                speed_joy = (self.joy.axes[2] - 1.0) / 2.0
            else:
                speed_joy = -1.0 * (self.joy.axes[5] - 1.0) / 2.0
            action = np.array([speed_joy, steer_joy])
        
        if self.emergency_stop:
            print(colored("Emergency stop", "red"))
            action = np.array([0., 0.])
            
        
        # Feed history to MPPI
        #  append history here because we sometimes wants to overwirte the mppi action 
        #  with other controllers (e.g. pure pursuit)
        if self.controller_type == 'mppi':
            self.mppi_running_params = self.mppi.feed_hist(self.mppi_running_params, state, action)
    

        action_rate = action - self.prev_action
        self.prev_action = action
        
        self._counter += 1
        
        # px, py, psi, vx, vy, omega = env.obs_state().tolist()
        px, py, psi, vx, vy, omega = state.tolist()
        
        q = quaternion_from_euler(0, 0, psi)
        now = self.get_clock().now().to_msg()
        cmd = AckermannDriveStamped()
        cmd.header.stamp = now
        cmd.drive.speed = float(action[0])
        cmd.drive.steering_angle = float(action[1])
        
        odom_copy.header.stamp = now

        pose_with_covariance_stamped = PoseWithCovarianceStamped()
        pose_with_covariance_stamped.header.frame_id = 'map'
        pose_with_covariance_stamped.header.stamp = now
        pose_with_covariance_stamped.pose.pose.position.x = px
        pose_with_covariance_stamped.pose.pose.position.y = py
        pose_with_covariance_stamped.pose.pose.orientation.x = q[0]
        pose_with_covariance_stamped.pose.pose.orientation.y = q[1]
        pose_with_covariance_stamped.pose.pose.orientation.z = q[2]
        pose_with_covariance_stamped.pose.pose.orientation.w = q[3]
        
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = now
        for i in range(target_pos_list.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(target_pos_list[i][0])
            pose.pose.position.y = float(target_pos_list[i][1])
            path.poses.append(pose)
        
        if self.controller_type == 'mppi':
            mppi_path = Path()
            mppi_path.header.frame_id = 'map'
            mppi_path.header.stamp = now
            for i in range(len(sampled_traj)):
                pose = PoseStamped()
                pose.header.frame_id = 'map'
                pose.pose.position.x = float(sampled_traj[i, 0])
                pose.pose.position.y = float(sampled_traj[i, 1])
                mppi_path.poses.append(pose)
        
        throttle = Float64()
        throttle.data = float(action[0]) * 3905.9 * 2
        
        steer = Float64()
        steer.data = float(action[1] * -1.0 / 2 + 0.5)
        
        action_rate_msg = Float64()
        action_rate_msg.data = float(np.linalg.norm(action_rate))
        self.action_rate_pub.publish(action_rate_msg)
        
        lateral_error = Float64()
        lateral_error.data = frenet_pose.t
        
        ref_vel_msg = deepcopy(self.odom)
        ref_vel_msg.twist.twist.linear.x = ref_vel
        
        # body polygon
        pts = np.array([
            [self.model_params.LF, self.L/3],
            [self.model_params.LF, -self.L/3],
            [-self.model_params.LR, -self.L/3],
            [-self.model_params.LR, self.L/3],
        ])
        # transform to world frame
        R = euler_matrix(0, 0, psi)[:2, :2]
        pts = np.dot(R, pts.T).T
        pts += np.array([px, py])
        body = PolygonStamped()
        body.header.frame_id = 'map'
        body.header.stamp = now
        for i in range(pts.shape[0]):
            p = Point32()
            p.x = float(pts[i, 0])
            p.y = float(pts[i, 1])
            p.z = 0.
            body.polygon.points.append(p)
        self.body_pub_.publish(body)

        end_time = self.get_clock().now()
        duration_sec = (end_time - start_time).nanoseconds / 1e9
        print(colored(f"duration: {duration_sec:.3f}", "red"))
        if duration_sec > self.model_params.DT:
            # print("Out of time!")
            self.get_logger().warn(f"MPPI took {duration_sec} seconds which is longer than DT {self.model_params.DT} seconds.", throttle_duration_sec=1.0)
        else:
            sleep_time = self.model_params.DT - duration_sec
            time.sleep(sleep_time)

        self.vehicle_cmd_pub_.publish(cmd)
        self.odom_copy_pub_.publish(odom_copy)
        # print(cmd.header.stamp, odom_copy.header.stamp)
        self.pose_pub_.publish(pose_with_covariance_stamped)
        self.ref_trajectory_pub_.publish(path)
        if self.controller_type == 'mppi':
            self.path_pub_.publish(mppi_path)
        self.throttle_pub_.publish(throttle)
        self.steer_pub_.publish(steer)
        self.lateral_error_pub_.publish(lateral_error)
        self.ref_vel_pub_.publish(ref_vel_msg)
        for i, (param, val) in enumerate(self.model_params.to_dict().items()):
            msg = Float64()
            msg.data = float(val)
            self.params_pub_list[i].publish(msg)
            
        if self.is_recover_mode:
            controller_type = "recover: pure pursuit"
        else:
            controller_type = self.controller_type
        misc_msg = String()
        misc_msg.data = f"env: {self.env_params.name}\ncontroller: {controller_type}\nEnv:\n- mass:{self.env_params.mass}\n- friction:{self.env_params.friction}\n- delay:{self.env_params.delay}\n- step:{self._counter}\n"
        self.misc_pub_.publish(misc_msg)
            
        #publish mppi time
        mppi_time_msg = Float64()
        mppi_time_msg.data = duration_sec
        self.mppi_time_pub_.publish(mppi_time_msg)
 
           
    def slow_timer_callback(self):
        # publish waypoint_list as path
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for i in range(self.global_planner.waypoints.shape[1]):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(self.global_planner.waypoints[0][i])
            pose.pose.position.y = float(self.global_planner.waypoints[1][i])
            path.poses.append(pose)
        self.waypoint_list_pub_.publish(path)

    def odom_callback(self, msg:Odometry):
        self.odom = msg
        if self.step_mode_:
            self.timer_callback()

    def joy_callback(self, msg):
        self.joy = msg
        if self.step_mode_:
            self.timer_callback()

def main():
    rclpy.init()
    car_node = CarNode()
    rclpy.spin(car_node)
    car_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
