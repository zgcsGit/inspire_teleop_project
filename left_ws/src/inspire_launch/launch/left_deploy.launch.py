from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    log_level_arg = DeclareLaunchArgument('log_level', default_value='info')
    robot_ip_arg = DeclareLaunchArgument('robot_ip', default_value='12.1.1.5')

    log_level = LaunchConfiguration('log_level')
    robot_ip = LaunchConfiguration('robot_ip')

    # =========================================
    # 1) Left Franka control node
    # =========================================
    franka_node = Node(
        package='franka_control',
        executable='franka_node',
        name='franka',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
        parameters=[{'robot_ip': robot_ip}]
    )

    # =========================================
    # 2) Left hand modbus node
    # =========================================
    hand_modbus_node = Node(
        package='inspire_hand_modbus',
        executable='inspire_hand_modbus_topic',
        name='hand_modbus',
        namespace='left',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
    )

    hand_modbus_delayed = TimerAction(
        period=2.0,
        actions=[hand_modbus_node]
    )

    # =========================================
    # 3) Left hand init commands
    # =========================================
    set_force = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--once',
            '/left/set_force_data',
            'inspire_interfaces/msg/SetForce1',
            '{finger_ids:[1,2,3,4,5,6],forces:[1000,1000,1000,1000,1000,1000]}'
        ],
        output='screen',
    )

    set_speed = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--once',
            '/left/set_speed_data',
            'inspire_interfaces/msg/SetSpeed1',
            '{finger_ids:[1,2,3,4,5,6],speeds:[1000,1000,1000,1000,1000,1000]}'
        ],
        output='screen',
    )

    set_angle = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--once',
            '/left/set_angle_data',
            'inspire_interfaces/msg/SetAngle1',
            '{finger_ids:[1,2,3,4,5,6],angles:[546,819,847,726,998,0]}'
        ],
        output='screen',
    )

    hand_init_force = TimerAction(period=2.1, actions=[set_force])
    hand_init_speed = TimerAction(period=2.3, actions=[set_speed])
    hand_init_angle = TimerAction(period=2.5, actions=[set_angle])

    return LaunchDescription([
        log_level_arg,
        robot_ip_arg,

        franka_node,

        hand_modbus_delayed,
        hand_init_force,
        hand_init_speed,
        hand_init_angle,
    ])