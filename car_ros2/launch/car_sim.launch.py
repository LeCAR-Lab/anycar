import os
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def get_share_file(package_name, *args):
    return os.path.join(get_package_share_directory(package_name), *args)


def get_sim_time_launch_arg():
    use_sim_time = LaunchConfiguration("use_sim_time")

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time", default_value="False", description="Use simulation clock if True"
    )

    return declare_use_sim_time_cmd, {"use_sim_time": use_sim_time}


def generate_launch_description():
    declare_use_sim_time_cmd, use_sim_time = get_sim_time_launch_arg()

    return LaunchDescription(
        [
            declare_use_sim_time_cmd,
            Node(
                package="car_ros2",
                executable="car_node",
                name="car_node",
                output="screen",
                parameters=[
                    use_sim_time,
                    {
                        "step_mode": True,
                    }
                ],
                remappings=[
                ],
                emulate_tty=True,
            ),
            Node(
                package="car_ros2",
                executable="car_simulator_node",
                name="car_simulator_node",
                output="screen",
                parameters=[
                    use_sim_time
                ],
                remappings=[
                ],
                emulate_tty=True,
            ),
        ]
    )
