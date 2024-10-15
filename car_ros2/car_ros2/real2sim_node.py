from copy import deepcopy
import rclpy
from rclpy.node import Node
import time

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point, PolygonStamped, Point32
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from std_msgs.msg import Float64
from visualization_msgs.msg import MarkerArray, Marker
from ackermann_msgs.msg import AckermannDriveStamped

from tf_transformations import quaternion_from_euler, euler_matrix, euler_from_quaternion

from car_dynamics.models_jax import DynamicBicycleModel
from car_dynamics.controllers_jax import MPPIController, rollout_fn_select
from car_dynamics.controllers_jax import WaypointGenerator
from car_dynamics.models_jax import ParamAdaptModel, AdaptDataset
import threading
from datetime import datetime
import numpy as np
import jax
import jax.numpy as jnp
from multiprocessing.pool import ThreadPool
from car_dynamics.models_torch import DynamicsTorch
import matplotlib.pyplot as plt

from car_ros2 import CAR_ROS2_TMP

from termcolor import colored
print("DEVICE", jax.devices())

from car_ros2.utils import load_dynamic_params, load_mppi_params, load_env_params, rospy_time_datatime_float

import matplotlib.pyplot as plt

trajectory_type = "counter oval"
SPEED = 1.



def fn():
    ...

class Real2SimNode(Node):
    def __init__(self):
        super().__init__('car_node')
        self.env_params = load_env_params(self)
        self.model_params = load_dynamic_params(self)
        self.mppi_params = load_mppi_params(self)
        self.model_params_adapt = deepcopy(self.model_params)
        
        
        param_config = {
            'mu': [0.5, 1.0],
            'B': [20., 20.],
            'C': [1., 3.],
            'MASS': [4.65, 4.65],
            'Iz': [.04, .1],
            'Ta': [16., 16.],
            'Tb': [-.0, .0],
            'Sa': [0.36, .36],
            'Sb': [-.0, .0],
            'hcom': [0.07, 0.13],
            'fr': [0.05, 0.11],
            'LF': [0.21, 0.21],
            'LR': [0.1, 0.1],
        }
        
        self.param_config = param_config
        
        DYNAMICS = self.mppi_params.dynamics
        self.L = self.model_params.LF + self.model_params.LR
        
        if DYNAMICS == 'dbm':
            self.dynamics = DynamicBicycleModel(self.model_params)
            self.dynamics.reset()
            self.rollout_fn = rollout_fn_select('dbm', self.dynamics, self.model_params.DT, self.L, self.model_params.LR)
        elif DYNAMICS == 'transformer':
            self.dynamics = DynamicsTorch({})
            print(colored("Loaded transformer model", "green"))
            print(colored(type(self.dynamics), "blue"))
            self.rollout_fn = rollout_fn_select('transformer', self.dynamics, self.model_params.DT, self.L, self.model_params.LR)
            
        self.key = jax.random.PRNGKey(0)
        key, self.key = jax.random.split(self.key)
        
        self.mppi = MPPIController(
            self.mppi_params, self.rollout_fn, fn, key
        )
        self.mppi_running_params = self.mppi.get_init_params()
        
        self.step_mode_ = self.declare_parameter('step_mode', False).value
        
        # self.mppi_running_params = self.mppi.get_init_params()

        self.path_pub_ = self.create_publisher(Path, 'path', 1)
        self.waypoint_list_pub_ = self.create_publisher(Path, 'waypoint_list', 1)
        self.ref_trajectory_pub_ = self.create_publisher(Path, 'ref_trajectory', 1)
        # self.pose_pub_ = self.create_publisher(PoseWithCovarianceStamped, 'pose', 1)
        # if not self.step_mode_:
        #     self.timer_ = self.create_timer(0.05, self.timer_callback)
        self.slow_timer_ = self.create_timer(1.0, self.slow_timer_callback)
        # self.throttle_pub_ = self.create_publisher(float32, 'speed', 1)
        # self.steer_pub_ = self.create_publisher(float32, 'steering', 1)
        # self.trajectory_array_sub_ = self.create_subscription(MarkerArray, 'trajectory_array', self.trajectory_callback, 1)
        # self.vehicle_cmd_pub_ = self.create_publisher(AckermannDriveStamped, 'ackermann_command', self.command_callback, 1)
        self.odom_sub_ = self.create_subscription(Odometry, 'odometry', self.odom_callback, 50)
        self.action_sub_ = self.create_subscription(AckermannDriveStamped, 'ackermann_command', self.action_callback, 50)
        # self.action_rate_pub = self.create_publisher(float32, 'debug/action_rate', 1)
        self.odom_list = []
        self.debug_real_trajectory_ = self.create_publisher(Path, 'debug_vis/real_path', 1)
        self.debug_rollout_trajectory = self.create_publisher(Path, 'debug_vis/rollout_path', 1)
        self.body_pub_ = self.create_publisher(PolygonStamped, 'debug_vis/body', 1)
        self.lr_pub_ = self.create_publisher(Float64, 'debug_vis/LR', 1)
        self.error_pub_ = self.create_publisher(Float64, 'debug_vis/predict_error', 1)
        self.loss_pub_ = self.create_publisher(Float64, 'debug/adapt/loss', 1)
        
        self.params_pub_list = []
        for param in self.model_params.to_dict().keys():
            self.params_pub_list.append(self.create_publisher(Float64, f"debug/param/{param}", 1))
            
        self.predict_H = self.mppi_params.num_intermediate * self.mppi_params.h_knot
        
        self.debug_odom_time = []
        self.debug_action_time = []
        
        self.state_list = []
        self.action_list = []
        
        self.is_terminate = False
        
        timer_thread = threading.Thread(target=self.timer_thread_fn)
        timer_thread.start()
        
        print("start timer thread")
        
        
        # adaptation_thread = threading.Thread(target=self.adaptation_thread_fn)
        # adaptation_thread.start()
        
    def timer_thread_fn(self):
        while True:
            # self.timer_callback()
            self.timer_debug_callback()
            if self.is_terminate:
                break
            

    def timer_debug_callback(self):
        if len(self.debug_action_time) < 50:
            print("len", len(self.debug_action_time), len(self.debug_odom_time))
        # if len(self.debug_action_time) > 0:
        #     print(self.debug_action_time[0])
        if len(self.debug_action_time) == 50:
            print("Print")
            action_time_list = np.array(rospy_time_datatime_float(self.debug_action_time))
            odom_time_list = np.array(rospy_time_datatime_float(self.debug_odom_time))
            start_time = min(min(action_time_list), np.min(odom_time_list))
            action_time_list -= start_time
            odom_time_list -= start_time
            print(action_time_list, odom_time_list)
            
            plt.scatter(np.arange(action_time_list.shape[0]), action_time_list, label="action", s=1, marker='x')
            plt.scatter(np.arange(odom_time_list.shape[0]), odom_time_list, label="odom", s=1, marker='o')
            plt.legend()
            plt.savefig(f"{CAR_ROS2_TMP}/debug_time.png")
            self.is_terminate = True
            

    def timer_callback(self):

        start_time = self.get_clock().now()
        
        if len(self.odom_list) < self.predict_H or self.odom_list[-self.predict_H] is None or \
            len(self.action_list) < self.predict_H*2+self.mppi_params.delay:
            return
        
        odom = self.odom_list[-self.predict_H*2]
        rpy = euler_from_quaternion([
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w,
        ])
        
        state = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, rpy[2], odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z])
        
        
        # print("odom time", rospy_time_to_datetime(self.debug_odom_time[-self.predict_H*2::2]))
        # print("action time", rospy_time_to_datetime(self.debug_action_time[-self.predict_H:]))
        
        prev_pos = np.array([[odom.pose.pose.position.x, odom.pose.pose.position.y] for odom in self.odom_list[-self.predict_H*2::2]])
        action_prev = np.array(self.action_list[-self.predict_H-self.mppi_params.delay:len(self.action_list)-self.mppi_params.delay])
        # action_prev[:, 1] = 1.
        # print(action_prev)
        action_prev_jnp = jnp.array(action_prev)
        
        # distance_list = np.linalg.norm(waypoint_list - obs[:2], axis=-1)
        # # import pdb; pdb.set_trace()
        # t_idx = np.argmin(distance_list)
        # t_closed = waypoint_t_list[t_idx]
        # target_pos_list = [reference_traj(0. + t_closed + i*DT*1.) for i in range(H+0+1)]
        # target_pos_tensor = jnp.array(target_pos_list)
        # target_pos_tensor = self.waypoint_generator.generate(jnp.array(state[:5]))
        
        # target_pos_list = np.array(target_pos_tensor)
        if hasattr(self.dynamics, "reset"):
            self.dynamics.reset()
        # dynamic_params_tuple = (self.model_params.LF, self.model_params.LR, self.model_params.MASS, self.model_params.DT, self.model_params.K_RFY, self.model_params.K_FFY, self.model_params.Iz, self.model_params.Ta, self.model_params.Tb, self.model_params.Sa, self.model_params.Sb, self.model_params.mu, self.model_params.Cf, self.model_params.Cr, self.model_params.Bf, self.model_params.Br, self.model_params.hcom, self.model_params.fr)
        
        predicted_H_trajectory = self.mppi.debug_rollout(state, action_prev_jnp, self.mppi_running_params)
        predicted_H_trajectory = np.array(predicted_H_trajectory[:-1])
        
        self.state_list.append(state)
        self.action_list.append(action_prev[0])
        
        predict_error = np.linalg.norm(predicted_H_trajectory[-1, :2] - prev_pos[-1])
        
        self.mppi.feed_hist(state, action_prev[0])
        # print("predict_error", predict_error)
    
        # print(predicted_H_trajectory[:, :2], prev_pos)
        # print(predicted_H_trajectory.shape)
        # st = time.time()
        # action, self.mppi_running_params, mppi_info = self.mppi(state,target_pos_tensor,self.mppi_running_params, vis_optim_traj=True)
        # print("time to compute action", time.time() - st)
        # st_ = time.time()
        # action = np.array(action)
        
        # action_rate = action - self.prev_action
        # self.prev_action = action
        # sampled_traj = np.array(mppi_info['trajectory'][:, :2])  

        # all_trajectory = np.array(mppi_info['all_traj'])[:, :, :2]
        # print("time to convert action to np", time.time() - st_)      
        # print("new obs", env.obs_state())

        
        # px, py, psi, vx, vy, omega = env.obs_state().tolist()
        px, py, psi, vx, vy, omega = state.tolist()
        
        q = quaternion_from_euler(0, 0, psi)
        now = self.get_clock().now().to_msg()

        # cmd = AckermannDriveStamped()
        # cmd.header.stamp = now
        # cmd.drive.speed = float(action[0])
        # cmd.drive.steering_angle = float(action[1])

        pose_with_covariance_stamped = PoseWithCovarianceStamped()
        pose_with_covariance_stamped.header.frame_id = 'map'
        pose_with_covariance_stamped.header.stamp = now
        pose_with_covariance_stamped.pose.pose.position.x = px
        pose_with_covariance_stamped.pose.pose.position.y = py
        pose_with_covariance_stamped.pose.pose.orientation.x = q[0]
        pose_with_covariance_stamped.pose.pose.orientation.y = q[1]
        pose_with_covariance_stamped.pose.pose.orientation.z = q[2]
        pose_with_covariance_stamped.pose.pose.orientation.w = q[3]
        
        pred_path = Path()
        pred_path.header.frame_id = 'map'
        pred_path.header.stamp = now
        for i in range(predicted_H_trajectory.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(predicted_H_trajectory[i][0])
            pose.pose.position.y = float(predicted_H_trajectory[i][1])
            pred_path.poses.append(pose)
        
        real_path = Path()
        real_path.header.frame_id = 'map'
        real_path.header.stamp = now
        for i in range(prev_pos.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(prev_pos[i, 0])
            pose.pose.position.y = float(prev_pos[i, 1])
            real_path.poses.append(pose)
            
        # print(predicted_H_trajectory.shape, prev_pos.shape)
        # rollout_path = Path()
        
        # throttle = float32()
        # throttle.data = float(action[0]) * 3905.9 * 2
        
        # steer = float32()
        # steer.data = float(action[1] * -1.0 / 2 + 0.5)
        
        # action_rate_msg = float32()
        # action_rate_msg.data = float(np.linalg.norm(action_rate))
        # self.action_rate_pub.publish(action_rate_msg)
        
        # trajectory array
        # all_trajectory is of shape horizon, num_rollout, 3
        # trajectory_array = MarkerArray()
        # for i in range(100):
        #     marker = Marker()
        #     marker.header.frame_id = 'map'
        #     marker.header.stamp = now
        #     marker.type = Marker.LINE_STRIP
        #     marker.action = Marker.ADD
        #     marker.id = i
        #     marker.scale.x = 0.05
        #     marker.color.a = 1.0
        #     marker.color.r = 1.0
        #     marker.color.g = 0.0
        #     marker.color.b = 0.0
        #     for j in range(all_trajectory.shape[0]):
        #         point = all_trajectory[j, i]
        #         p = Point()
        #         p.x = float(point[0])
        #         p.y = float(point[1])
        #         p.z = 0.
        #         marker.points.append(p)
        #     trajectory_array.markers.append(marker)
        # self.trajectory_array_pub_.publish(trajectory_array)
        
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
        if duration_sec > self.model_params.DT:
            self.get_logger().warn(f"MPPI took {duration_sec} seconds which is longer than DT {self.model_params.DT} seconds.", throttle_duration_sec=1.0)
        else:
            sleep_time = self.model_params.DT - duration_sec
            time.sleep(sleep_time)

        # self.vehicle_cmd_pub_.publish(cmd)
        # self.pose_pub_.publish(pose_with_covariance_stamped)
        # self.ref_trajectory_pub_.publish(path)
        # self.path_pub_.publish(mppi_path)
        # self.throttle_pub_.publish(throttle)
        # self.steer_pub_.publish(steer)
        self.debug_rollout_trajectory.publish(pred_path)
        self.debug_real_trajectory_.publish(real_path)
        
        msg = Float64()
        msg.data = float(predict_error)
        self.error_pub_.publish(msg)
        
        for i, (param, val) in enumerate(self.model_params.to_dict().items()):
            msg = Float64()
            msg.data = float(val)
            self.params_pub_list[i].publish(msg)
    
    def adaptation_thread_fn(self):
        while True:
            self.adaptation_callback()
    
    def adaptation_callback(self):
        """
            Take in state and action pair
            feed to MLP to do gradient descent for few rounds, then update the dynamics model
        """
        # print(f"Start to adapt {len(self.state_list)} samples")
        if len(self.state_list) < self.model_params_adapt.num_envs + self.mppi_params.delay + 2:
            print("Before Adapt")
            time.sleep(self.model_params_adapt.DT * 2.)
            return
        
        start_time = time.time()
        state_list = self.state_list[-self.model_params_adapt.num_envs-1:]
        action_list = self.action_list[-self.model_params_adapt.num_envs-1-self.mppi_params.delay:len(self.action_list)-self.mppi_params.delay]
        state_jnp = jnp.array(state_list)
        action_jnp = jnp.array(action_list[:len(action_list)-self.mppi_params.delay])
        
        dataset = AdaptDataset(state_list=state_jnp, action_list=action_jnp)
        # init_params = jnp.array([self.model_params.mu, self.model_params.Bf, self.model_params.Cf])
        # init_mu = np.random.uniform(0.5, 1.)
        # init_B = np.random.uniform(1., 30.)
        # init_C = np.random.uniform(1., 5.)
        # init_MASS = np.random.uniform(1., 10.)
        # init_Iz = np.random.uniform(.001, 1.)
        # init_Ta = np.random.uniform(1., 20.)
        # init_Tb = np.random.uniform(-5., 5.)
        # init_Sa = np.random.uniform(0.1, .4)
        # init_Sb = np.random.uniform(-.2, .2)
        # init_hcom = np.random.uniform(0.01, 0.3)
        # init_fr = np.random.uniform(0.001, 0.1)
        
        # declare a dict, with keys as param_name (mu, B, C, ...), value to be list [min, max] for example {'mu': [0.5, 1.0]}

        
        # init_params = jnp.array([init_mu, init_B, init_C, init_MASS, init_Iz, init_Ta, init_Tb, init_Sa, init_Sb, init_hcom, init_fr])
        
        # init_params = jnp.array([np.random.uniform(-100, 100) for _ in range(13)])
        min_loss = 100.
        min_params = None
        
        def search_param(i):
            adapt_params = jnp.array([self.model_params.mu, self.model_params.Bf, self.model_params.Cf, self.model_params.MASS, self.model_params.Iz, self.model_params.Ta, self.model_params.Tb, self.model_params.Sa, self.model_params.Sb, self.model_params.hcom, self.model_params.fr, self.model_params.LF, self.model_params.LR])
            adapt_param_min = jnp.array([self.param_config[key][0] for key in ['mu', 'B', 'C', 'MASS', 'Iz', 'Ta', 'Tb', 'Sa', 'Sb', 'hcom', 'fr', 'LF', 'LR']])
            adapt_param_max = jnp.array([self.param_config[key][1] for key in ['mu', 'B', 'C', 'MASS', 'Iz', 'Ta', 'Tb', 'Sa', 'Sb', 'hcom', 'fr', 'LF', 'LR']])
            adapt_params = (adapt_params - adapt_param_min) / (adapt_param_max - adapt_param_min + 1e-5)
            
            # adapt_params = np.random.uniform(0.00, 1., 13)
            init_uniform = adapt_params
            init_uniform = np.log(init_uniform / (1. - init_uniform + 1e-5))
            init_params = jnp.array(init_uniform)
            params, adapt_info = self.adapter.adapt(dataset, init_params, 10)    
            return adapt_info['loss'], adapt_info['loss_all'], params

        with ThreadPool() as pool:
            result = pool.map(search_param, range(1))
            result = sorted(result, key=lambda x: x[0])
            
        min_loss, loss_all, params = result[0]
        # print()
        end_time = time.time()
        duration_sec = end_time - start_time
        
        print(colored(f"params: {params}", "yellow"))
        print(colored(f"Adaptation loss per epoch: {loss_all}", "red"))
        
        if duration_sec > self.model_params.DT * 2.:
            # self.get_logger().warn(f"MPPI took {duration_sec} seconds which is longer than DT {self.model_params.DT} seconds.", throttle_duration_sec=1.0)
            pass
        else:
            sleep_time = self.model_params.DT * 2. - duration_sec
            time.sleep(sleep_time)
            
        # self.adapt_alpha = 1.
        if min_loss > 0.1:
            adapt_alpha = 0.
        else:
            adapt_alpha = self.adapt_alpha
        self.model_params.mu = float(params[0]) * adapt_alpha + self.model_params.mu * (1. - adapt_alpha)
        self.model_params.Bf = float(params[1]) * adapt_alpha + self.model_params.Bf * (1. - adapt_alpha)
        self.model_params.Br = float(params[1]) * adapt_alpha + self.model_params.Br * (1. - adapt_alpha)
        self.model_params.Cf = float(params[2]) * adapt_alpha + self.model_params.Cf * (1. - adapt_alpha)
        self.model_params.Cr = float(params[2]) * adapt_alpha + self.model_params.Cr * (1. - adapt_alpha)
        self.model_params.MASS = float(params[3]) * adapt_alpha + self.model_params.MASS * (1. - adapt_alpha)
        self.model_params.Iz = float(params[4]) * adapt_alpha + self.model_params.Iz * (1. - adapt_alpha)
        self.model_params.Ta = float(params[5]) * adapt_alpha + self.model_params.Ta * (1. - adapt_alpha)
        self.model_params.Tb = float(params[6]) * adapt_alpha + self.model_params.Tb * (1. - adapt_alpha)
        self.model_params.Sa = float(params[7]) * adapt_alpha + self.model_params.Sa * (1. - adapt_alpha)
        self.model_params.Sb = float(params[8]) * adapt_alpha + self.model_params.Sb * (1. - adapt_alpha)
        self.model_params.hcom = float(params[9]) * adapt_alpha + self.model_params.hcom * (1. - adapt_alpha)
        self.model_params.fr = float(params[10]) * adapt_alpha + self.model_params.fr * (1. - adapt_alpha)
        self.model_params.LF = float(params[11]) * adapt_alpha + self.model_params.LF * (1. - adapt_alpha)
        self.model_params.LR = float(params[12]) * adapt_alpha + self.model_params.LR * (1. - adapt_alpha)
        
        
        # print(params, min_loss,)    
        
        msg = Float64()
        msg.data = float(min_loss)
        self.lr_pub_.publish(msg)
        
        msg = Float64()
        msg.data = float(min_loss)
        self.loss_pub_.publish(msg)
        
    def slow_timer_callback(self):
        # publish waypoint_list as path
        # path = Path()
        # path.header.frame_id = 'map'
        # path.header.stamp = self.get_clock().now().to_msg()
        # for i in range(self.waypoint_generator.waypoint_list_np.shape[0]):
        #     pose = PoseStamped()
        #     pose.header.frame_id = 'map'
        #     pose.pose.position.x = float(self.waypoint_generator.waypoint_list_np[i][0])
        #     pose.pose.position.y = float(self.waypoint_generator.waypoint_list_np[i][1])
        #     path.poses.append(pose)
        # self.waypoint_list_pub_.publish(path)
        pass

    def odom_callback(self, msg:Odometry):
        self.odom_list.append(msg)
        self.debug_odom_time.append(msg.header.stamp)
        # self.odom_list = self.odom_list[-self.predict_H*10:]
        # if self.step_mode_:
        #     self.timer_callback()
    
    def action_callback(self, msg:AckermannDriveStamped):
        action0 = msg.drive.speed
        action1 = msg.drive.steering_angle      
        self.debug_action_time.append(msg.header.stamp)  
        self.action_list.append([action0, action1])
        # self.action_list = self.action_list[-self.predict_H*10:]
        # if self.step_mode_:
        #     self.timer_callback()
            
def main():
    rclpy.init()
    car_node = Real2SimNode()
    rclpy.spin(car_node)
    car_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
