from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    urdf_file = os.path.join(get_package_share_directory('autonomous_tb3'), 'urdf', 'mecanum_robot.urdf')
    
    return LaunchDescription([
        # Launch Gazebo
        Node(
            package='gazebo_ros',
            executable='gzserver',
            output='screen',
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': open(urdf_file).read()}]
        ),
    ])
