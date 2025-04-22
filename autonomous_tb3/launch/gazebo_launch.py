import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    urdf_file = '/home/souri/ros2_ws/src/ROS2-Autonomous-Driving-and-Navigation-SLAM-with-TurtleBot3/autonomous_tb3/urdf/mecanum_robot.urdf'

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'),

        ExecuteProcess(
            cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so'],
            output='screen'),

        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-entity', 'mecanum_robot', '-file', urdf_file],
            output='screen'),
    ])
