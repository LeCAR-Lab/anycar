from setuptools import find_packages, setup
import os
import glob

package_name = 'car_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob.glob('launch/*.py')),
        (os.path.join('share', package_name), glob.glob('param/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lecar',
    maintainer_email='randyxiao64@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'car_node = car_ros2.car_node:main',
            'real2sim_node = car_ros2.real2sim_node:main',
            'car_simulator_node = car_ros2.car_simulator_node:main',
            'car_joystick_node = car_ros2.car_joystick_node:main',
            'car_slam_node = car_ros2.car_slam_node:main',
            'car_vio_node = car_ros2.car_vio_node:main'
        ],
    },
)
