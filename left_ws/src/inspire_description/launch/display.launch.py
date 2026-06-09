from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    hand_dir = get_package_share_directory('inspire_description')  

    urdf_path = os.path.join(hand_dir, 'urdf', 'inspire_hand_left.urdf')
    rviz_config_path = os.path.join(hand_dir, 'config', 'display.rviz')

    # 读取URDF内容
    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    return LaunchDescription([
        # joint_state_publisher_gui，用来用滑条控制关节
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen',
            remappings=[
                ('/joint_states', '/left_hand/joint_states')
            ],
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
            remappings=[
                ('/joint_states', '/left_hand/joint_states')
            ],
        ),
        
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d',rviz_config_path] 
        )
    ])
