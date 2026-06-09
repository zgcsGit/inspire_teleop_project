import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, IncludeLaunchDescription, ExecuteProcess
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # --------------------------
    # Args
    # --------------------------
    log_level_arg = DeclareLaunchArgument('log_level', default_value='info')
    robot_ip_arg  = DeclareLaunchArgument('robot_ip', default_value='12.1.1.6')
    modbus_ip_arg = DeclareLaunchArgument('modbus_ip', default_value='192.168.11.210')
    modbus_port_arg = DeclareLaunchArgument('modbus_port', default_value='6000')
    desired_pose_topic_arg = DeclareLaunchArgument(
        'desired_pose_topic',
        default_value='/right/desired_pose_matrix'
    )
    actual_ee_pose_topic_arg = DeclareLaunchArgument(
        'actual_ee_pose_topic',
        default_value='/frankaRight/ee_pose_matrix'
    )
    input_hand_topic_arg = DeclareLaunchArgument(
        'input_hand_topic',
        default_value='/Righthandpoint'
    )

    # hand_ns controls BOTH:
    #   - modbus node namespace
    #   - init topics prefix: /<hand_ns>/set_...
    hand_ns_arg   = DeclareLaunchArgument('hand_ns', default_value='right')  # or 'left'

    log_level = LaunchConfiguration('log_level')
    robot_ip  = LaunchConfiguration('robot_ip')
    modbus_ip = LaunchConfiguration('modbus_ip')
    modbus_port = LaunchConfiguration('modbus_port')
    desired_pose_topic = LaunchConfiguration('desired_pose_topic')
    actual_ee_pose_topic = LaunchConfiguration('actual_ee_pose_topic')
    input_hand_topic = LaunchConfiguration('input_hand_topic')
    hand_ns   = LaunchConfiguration('hand_ns')

    # =========================================
    # 1) Franka launch (t = 0)
    # =========================================
    franka_launch_path = os.path.join(
        get_package_share_directory('franka_control'),
        'launch',
        'inspire_franka.launch.py'
    )
    franka_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(franka_launch_path),
        launch_arguments={
            'robot_ip': robot_ip,
            'log_level': log_level,
            'desired_pose_topic': desired_pose_topic,
            'actual_ee_pose_topic': actual_ee_pose_topic,
            'input_hand_topic': input_hand_topic,
        }.items()
    )

    # =========================================
    # 2) Hand modbus node (t = 2.0s)
    # =========================================
    hand_modbus_node = Node(
        package='inspire_hand_modbus',
        executable='inspire_hand_modbus_topic',
        name='hand_modbus',
        namespace=hand_ns,   # -> topics become /right/... or /left/...
        output='screen',
        parameters=[
            {'modbus_ip': modbus_ip},
            {'modbus_port': modbus_port},
        ],
        arguments=['--ros-args', '--log-level', log_level],
    )

    hand_modbus_delayed = TimerAction(
        period=2.0,
        actions=[hand_modbus_node]
    )

    # Helper: build "/<hand_ns>/xxx" as a Launch substitution
    def ns_topic(suffix: str):
        # suffix should start with '/', like '/set_force_data'
        return [TextSubstitution(text='/'), hand_ns, TextSubstitution(text=suffix)]

    # =========================================
    # 3) Hand init commands (t = 2.1s / 2.3s / 2.5s)
    # =========================================
    set_force = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--once',
            ns_topic('/set_force_data'),
            'inspire_interfaces/msg/SetForce1',
            '{finger_ids:[1,2,3,4,5,6],forces:[1000,1000,1000,1000,1000,1000]}'
        ],
        output='screen',
    )

    set_speed = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--once',
            ns_topic('/set_speed_data'),
            'inspire_interfaces/msg/SetSpeed1',
            '{finger_ids:[1,2,3,4,5,6],speeds:[1000,1000,1000,1000,1000,1000]}'
        ],
        output='screen',
    )

    set_angle = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--once',
            ns_topic('/set_angle_data'),
            'inspire_interfaces/msg/SetAngle1',
            '{finger_ids:[1,2,3,4,5,6],angles:[1000,1000,1000,1000,1000,1000]}'
        ],
        output='screen',
    )

    # 让 init 稍微晚于 modbus 启动（避免 pub 时 subscriber 还没起来）
    hand_init_force = TimerAction(period=2.1, actions=[set_force])
    hand_init_speed = TimerAction(period=2.3, actions=[set_speed])
    hand_init_angle = TimerAction(period=2.5, actions=[set_angle])

    return LaunchDescription([
        log_level_arg,
        robot_ip_arg,
        modbus_ip_arg,
        modbus_port_arg,
        desired_pose_topic_arg,
        actual_ee_pose_topic_arg,
        input_hand_topic_arg,
        hand_ns_arg,

        franka_launch,

        hand_modbus_delayed,

        hand_init_force,
        hand_init_speed,
        hand_init_angle,
    ])
