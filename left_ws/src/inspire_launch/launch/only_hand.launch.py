import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ---------- log level ----------
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level for all nodes'
    )
    log_level = LaunchConfiguration('log_level')

    # ---------- Node: inspire_hand_modbus ----------
    hand_modbus_node = Node(
        package='inspire_hand_modbus',
        executable='inspire_hand_modbus_topic',
        name='hand_modbus',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level]
    )
    
        # ---------- 发送 topic 命令, 初始化机械手 ----------
    set_force = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--once',
            '/set_force_data', 'inspire_interfaces/msg/SetForce1',
            '{finger_ids:[1,2,3,4,5,6],forces:[1000,1000,1000,1000,1000,1000]}'
        ]
    )

    set_speed = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--once',
            '/set_speed_data', 'inspire_interfaces/msg/SetSpeed1',
            '{finger_ids:[1,2,3,4,5,6],speeds:[1000,1000,1000,1000,1000,1000]}'
        ]
    )

    set_angle = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--once',
            '/set_angle_data', 'inspire_interfaces/msg/SetAngle1',
            '{finger_ids:[1,2,3,4,5,6],angles:[1000,1000,1000,1000,1000,1000]}'
        ]
    )

    # ---------- 返回 LaunchDescription ----------
    return LaunchDescription([
        log_level_arg,
        hand_modbus_node,
        set_force,
        set_speed,
        set_angle,
    ])
