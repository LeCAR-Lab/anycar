
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rich.progress import track
from scipy.spatial.transform import Rotation as R
import numpy as np
from car_ros2.utils import rospy_time_datatime_float
from tf_transformations import euler_from_quaternion

def read_messages(input_bag: str):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_bag, storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )

    topic_types = reader.get_all_topics_and_types()

    def typename(topic_name):
        for topic_type in topic_types:
            if topic_type.name == topic_name:
                return topic_type.type
        raise ValueError(f"topic {topic_name} not in bag")

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        # import ipdb; ipdb.set_trace()
        # print(topic)
        if topic == '/vehicle_state':
            msg = None
            timestamp = 0.
        elif topic == '/sensors/core':
            msg = None
            timestamp = 0.
        elif topic == '/vehicle/steering_report':
            msg = None
            timestamp = 0.
        elif topic == '/sensors/imu':
            msg = None
            timestamp = 0.
        elif topic == "/vehicle/engine_report":
            msg = None
            timestamp = 0.
        else:
            msg_type = get_message(typename(topic))
            msg = deserialize_message(data, msg_type)
        yield topic, msg, timestamp
    del reader
    
class BagReader:
    def __init__(self, bag_path, full_state=False):
        self.full_state = full_state # whether include orientation in state
        self.bag_path = bag_path
        self.bag_dict = {}
        for topic, msg, timestamp_unkown in read_messages(self.bag_path):
            # import pdb; pdb.set_trace()
            # timestamp = rospy_time_datatime_float(msg.stamp)
            timestamp, msg = self.parse_msg(topic, msg)
            if topic in self.bag_dict.keys():
                self.bag_dict[topic].append((timestamp, msg))
            else:
                self.bag_dict[topic] = [(timestamp, msg)]
        
        for key in self.bag_dict.keys():
            # sort list by timestamp
            if self.bag_dict[key][0][0] is None:
                continue
            # print()
            self.bag_dict[key].sort(key=lambda x: x[0])
            
        print("[INFO] Read bag, get topics:", self.bag_dict.keys())
        
        
    def parse_msg(self, topic, msg):
        # print(topic)
        if topic == '/ackermann_command':
            # import pdb; pdb.set_trace()
            timestamp = rospy_time_datatime_float(msg.header.stamp)
            return timestamp, np.array([msg.drive.speed, msg.drive.steering_angle])
        elif topic in ['/odometry', '/odometry_copy', '/odometry/vicon/filtered']:
            timestamp = rospy_time_datatime_float(msg.header.stamp)
            rpy = euler_from_quaternion([
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ])
            
            if self.full_state:
                state = np.array([msg.pose.pose.position.x,     # 0
                                msg.pose.pose.position.y,     # 1
                                msg.pose.pose.orientation.x,  # 2
                                msg.pose.pose.orientation.y,  # 3
                                msg.pose.pose.orientation.z,  # 4
                                msg.pose.pose.orientation.w,  # 5
                                msg.twist.twist.linear.x,     # 6
                                msg.twist.twist.linear.y,     # 7
                                msg.twist.twist.angular.z     # 8
                                ])
            else:
                state = np.array([msg.pose.pose.position.x,     # 0
                                msg.pose.pose.position.y,     # 1
                                rpy[2],                       # 2
                                msg.twist.twist.linear.x,     # 3
                                msg.twist.twist.linear.y,     # 4
                                msg.twist.twist.angular.z     # 5
                                ])
            
            return timestamp, state
        elif topic == '/vicon_pose':
            timestamp = rospy_time_datatime_float(msg.header.stamp)
            # /vicon_pose.pose.pose.position.x
            vicon_pos = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ])
            return timestamp, vicon_pos
        elif topic == '/speed':
            return None, msg.data
        elif topic == '/steering':
            return None, msg.data
        elif topic == '/ref_trajectory':
            # import ipdb; ipdb.set_trace()
            timestamp = rospy_time_datatime_float(msg.header.stamp)
            path = np.array([[pose.pose.position.x, pose.pose.position.y] for pose in msg.poses])
            return timestamp, path
        elif topic == '/lateral_error':
            return None, msg.data            
        elif topic == '/tracking/ref_vel':
            timestamp = rospy_time_datatime_float(msg.header.stamp)
            ref_vx = msg.twist.twist.linear.x
            return timestamp, ref_vx
        else:
            return None, None
    
def synchronize_time(t_list1, t_list2, state_list1):
    assert len(t_list1) >= len(t_list2)
    assert t_list1[0] <= t_list2[0]
    ptr1 = 0
    ptr2 = 0
    t_list1_new = []
    state_list1_new = []
    while ptr2 < len(t_list2):
        while t_list1[ptr1] <= t_list2[ptr2] and ptr1 < len(t_list1) - 1 and t_list1[ptr1+1] <= t_list2[ptr2]:
            ptr1 += 1
        t_list1_new.append(t_list1[ptr1])
        state_list1_new.append(state_list1[ptr1])
        ptr2 += 1
        
    t_list1_new = np.array(t_list1_new)
    state_list1_new = np.array(state_list1_new)
        
    return t_list1_new, state_list1_new