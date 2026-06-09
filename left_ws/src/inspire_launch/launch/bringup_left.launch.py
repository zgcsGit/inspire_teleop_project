import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, IncludeLaunchDescription, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def _make_venv_env(venv_path: str):
    """Build env overlay to run nodes inside a python venv while keeping ROS env."""
    ros_env = {
        'PATH': os.environ.get('PATH', ''),
        'PYTHONPATH': os.environ.get('PYTHONPATH', ''),
        'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', '')
    }

    env = ros_env.copy()
    env['PATH'] = os.path.join(venv_path, 'bin') + ':' + env['PATH']
    env['PYTHONPATH'] = os.path.join(venv_path, 'lib/python3.10/site-packages') + ':' + env['PYTHONPATH']
    return env


def generate_launch_description():
    # --------------------------
    # Args
    # --------------------------
    log_level_arg = DeclareLaunchArgument('log_level', default_value='info')
    robot_ip_arg  = DeclareLaunchArgument('robot_ip', default_value='12.1.1.5')
    pose_input_topic_arg = DeclareLaunchArgument('pose_input_topic', default_value='/Lefthandpoint')
    pose_output_topic_arg = DeclareLaunchArgument('pose_output_topic', default_value='/left/desired_pose_matrix')
    initial_frame_count_arg = DeclareLaunchArgument('initial_frame_count', default_value='21')
    alpha_translation_arg = DeclareLaunchArgument('alpha_translation', default_value='0.3')
    alpha_rotation_arg = DeclareLaunchArgument('alpha_rotation', default_value='0.2')
    t21_x_mm_arg = DeclareLaunchArgument('t21_x_mm', default_value='66.0')
    t21_y_mm_arg = DeclareLaunchArgument('t21_y_mm', default_value='0.0')
    t21_z_mm_arg = DeclareLaunchArgument('t21_z_mm', default_value='66.0')
    t0a_x_mm_arg = DeclareLaunchArgument('t0a_x_mm', default_value='405.0')
    t0a_y_mm_arg = DeclareLaunchArgument('t0a_y_mm', default_value='-441.0')
    t0a_z_mm_arg = DeclareLaunchArgument('t0a_z_mm', default_value='-90.0')
    modbus_ip_arg = DeclareLaunchArgument('modbus_ip', default_value='192.168.11.210')
    modbus_port_arg = DeclareLaunchArgument('modbus_port', default_value='6000')
    vicon_udp_ip_arg = DeclareLaunchArgument('vicon_udp_ip', default_value='0.0.0.0')
    vicon_udp_port_arg = DeclareLaunchArgument('vicon_udp_port', default_value='5005')
    vicon_filter_alpha_arg = DeclareLaunchArgument('vicon_filter_alpha', default_value='0.2')
    vicon_scale_m_arg = DeclareLaunchArgument('vicon_scale_m', default_value='0.001')
    dex_retargeting_robot_dir_arg = DeclareLaunchArgument(
        'dex_retargeting_robot_dir',
        default_value=os.path.expanduser(os.environ.get('DEX_RETARGETING_ROBOT_DIR', '~/dex-retargeting/assets/robots/hands'))
    )

    log_level = LaunchConfiguration('log_level')
    robot_ip  = LaunchConfiguration('robot_ip')
    pose_input_topic = LaunchConfiguration('pose_input_topic')
    pose_output_topic = LaunchConfiguration('pose_output_topic')
    initial_frame_count = LaunchConfiguration('initial_frame_count')
    alpha_translation = LaunchConfiguration('alpha_translation')
    alpha_rotation = LaunchConfiguration('alpha_rotation')
    t21_x_mm = LaunchConfiguration('t21_x_mm')
    t21_y_mm = LaunchConfiguration('t21_y_mm')
    t21_z_mm = LaunchConfiguration('t21_z_mm')
    t0a_x_mm = LaunchConfiguration('t0a_x_mm')
    t0a_y_mm = LaunchConfiguration('t0a_y_mm')
    t0a_z_mm = LaunchConfiguration('t0a_z_mm')
    modbus_ip = LaunchConfiguration('modbus_ip')
    modbus_port = LaunchConfiguration('modbus_port')
    vicon_udp_ip = LaunchConfiguration('vicon_udp_ip')
    vicon_udp_port = LaunchConfiguration('vicon_udp_port')
    vicon_filter_alpha = LaunchConfiguration('vicon_filter_alpha')
    vicon_scale_m = LaunchConfiguration('vicon_scale_m')
    dex_retargeting_robot_dir = LaunchConfiguration('dex_retargeting_robot_dir')

    # --------------------------
    # venv path
    # --------------------------
    finger_map_venv = os.path.expanduser(os.environ.get('FINGER_MAP_VENV', '~/finger_map_venv'))
    venv_env = _make_venv_env(finger_map_venv)

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
            'pose_input_topic': pose_input_topic,
            'pose_output_topic': pose_output_topic,
            'initial_frame_count': initial_frame_count,
            'alpha_translation': alpha_translation,
            'alpha_rotation': alpha_rotation,
            't21_x_mm': t21_x_mm,
            't21_y_mm': t21_y_mm,
            't21_z_mm': t21_z_mm,
            't0a_x_mm': t0a_x_mm,
            't0a_y_mm': t0a_y_mm,
            't0a_z_mm': t0a_z_mm,
        }.items()
    )

    # =========================================
    # 2) Hand modbus node for the left Inspire hand (t = 2.0s)
    # 注意：executable 仍然是 inspire_hand_modbus_topic（不改动）
    # 关键：namespace='left'
    # =========================================
    hand_modbus_node = Node(
        package='inspire_hand_modbus',
        executable='inspire_hand_modbus_topic',
        name='hand_modbus',
        namespace='left',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
        parameters=[{
            'modbus_ip': modbus_ip,
            'modbus_port': ParameterValue(modbus_port, value_type=int),
        }]
    )

    hand_modbus_delayed = TimerAction(
        period=2.0,
        actions=[hand_modbus_node]
    )

    # =========================================
    # 2.1) Hand init commands for the left Inspire hand
    # topic 全部加 /left 前缀
    # =========================================
    set_force = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--once',
            '/left/set_force_data', 'inspire_interfaces/msg/SetForce1',
            '{finger_ids:[1,2,3,4,5,6],forces:[1000,1000,1000,1000,1000,1000]}'
        ],
        output='screen',
    )

    set_speed = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--once',
            '/left/set_speed_data', 'inspire_interfaces/msg/SetSpeed1',
            '{finger_ids:[1,2,3,4,5,6],speeds:[1000,1000,1000,1000,1000,1000]}'
        ],
        output='screen',
    )

    set_angle = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--once',
            '/left/set_angle_data', 'inspire_interfaces/msg/SetAngle1',
            '{finger_ids:[1,2,3,4,5,6],angles:[1000,1000,1000,1000,1000,1000]}'
        ],
        output='screen',
    )

    hand_init_force = TimerAction(period=2.0, actions=[set_force])
    hand_init_speed = TimerAction(period=2.2, actions=[set_speed])
    hand_init_angle = TimerAction(period=2.4, actions=[set_angle])

    # =========================================
    # 3) Retargeting nodes with venv (顺序启动)
    # pub_csv -> finger_mapper -> vicon_udp
    # =========================================
    pub_csv_node = Node(
        package='inspire_retargeting',
        executable='pub_csv',
        name='pub_csv',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
        additional_env=venv_env,
        parameters=[{'dex_retargeting_robot_dir': dex_retargeting_robot_dir}],
    )

    # ---- finger_map: LEFT ----
    finger_mapper_left_node = Node(
        package='finger_map',
        executable='finger_mapper_node',
        name='finger_mapper_left',
        output='screen',
        arguments=[
            '--ros-args', '--log-level', log_level,
            '-p', 'input_topic:=/left_hand/joint_states',
            '-p', 'output_topic:=/left/set_angle_data',
            '-p', 'freeze_topic:=/teleop/left_hand_freeze',
            '-p', 'is_frozen_topic:=/left/hand_is_frozen',
            # 可选：平滑解冻参数
            '-p', 'ramp_duration:=1.0',
            # '-p', 'publish_on_freeze_edge:=true',
            # '-p', 'debug:=false',
        ],
        additional_env=venv_env,
    )

    # ---- finger_map: RIGHT ----
    finger_mapper_right_node = Node(
        package='finger_map',
        executable='finger_mapper_node',
        name='finger_mapper_right',
        output='screen',
        arguments=[
            '--ros-args', '--log-level', log_level,
            '-p', 'input_topic:=/right_hand/joint_states',
            '-p', 'output_topic:=/right/set_angle_data',
            '-p', 'freeze_topic:=/teleop/right_hand_freeze',
            '-p', 'is_frozen_topic:=/right/hand_is_frozen',
            # 可选：平滑解冻参数
            '-p', 'ramp_duration:=1.0',
            # '-p', 'publish_on_freeze_edge:=true',
            # '-p', 'debug:=false',
        ],
        additional_env=venv_env,
    )

    # 两个都启动（可以同一时刻，也可以错开 0.1s）
    start_finger_mapper_left  = TimerAction(period=3.2, actions=[finger_mapper_left_node])
    start_finger_mapper_right = TimerAction(period=3.3, actions=[finger_mapper_right_node])


    vicon_udp_node = Node(
        package='vicon_udp_receiver',
        executable='udp_lefthand_10',
        name='udp_lefthand_10',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
        parameters=[{
            'udp_ip': vicon_udp_ip,
            'udp_port': ParameterValue(vicon_udp_port, value_type=int),
            'filter_alpha': ParameterValue(vicon_filter_alpha, value_type=float),
            'scale_m': ParameterValue(vicon_scale_m, value_type=float),
        }]
    )

    start_pub_csv       = TimerAction(period=3.0, actions=[pub_csv_node])
    # start_finger_mapper = TimerAction(period=3.2, actions=[finger_mapper_node])
    start_vicon_udp     = TimerAction(period=3.4, actions=[vicon_udp_node])

    # =========================================
    # 4) Visualization launch at end
    # =========================================
    display_launch_path = os.path.join(
        get_package_share_directory('inspire_description'),
        'launch',
        'display.launch.py'
    )
    display_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(display_launch_path),
        launch_arguments={'log_level': log_level}.items()
    )

    start_display = TimerAction(period=4.0, actions=[display_launch])

    return LaunchDescription([
        log_level_arg,
        robot_ip_arg,
        pose_input_topic_arg,
        pose_output_topic_arg,
        initial_frame_count_arg,
        alpha_translation_arg,
        alpha_rotation_arg,
        t21_x_mm_arg,
        t21_y_mm_arg,
        t21_z_mm_arg,
        t0a_x_mm_arg,
        t0a_y_mm_arg,
        t0a_z_mm_arg,
        modbus_ip_arg,
        modbus_port_arg,
        vicon_udp_ip_arg,
        vicon_udp_port_arg,
        vicon_filter_alpha_arg,
        vicon_scale_m_arg,
        dex_retargeting_robot_dir_arg,

        franka_launch,

        # left hand Modbus node
        hand_modbus_delayed,

        # left hand initialization
        hand_init_force,
        hand_init_speed,
        hand_init_angle,

        # chain
        start_pub_csv,
        start_finger_mapper_left,
        start_finger_mapper_right,
        start_vicon_udp,

        # start_display,
    ])
