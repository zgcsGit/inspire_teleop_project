from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level for both nodes'
    )

    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='12.1.1.5',
        description='IP address of the Franka robot'
    )
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

    log_level = LaunchConfiguration('log_level')
    robot_ip = LaunchConfiguration('robot_ip')
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
    pkg_share = get_package_share_directory('franka_control_trans')
    rviz_config_path = os.path.join(pkg_share, 'config', 'config.rviz')

    franka_node = Node(
        package='franka_control',
        executable='franka_node',
        name='franka',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
        parameters=[{'robot_ip': robot_ip}]
    )

    pose_publisher_node = Node(
        package='franka_control_trans',
        executable='pose_publisher',
        name='pose_publisher',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
        parameters=[{
            'input_topic': pose_input_topic,
            'output_topic': pose_output_topic,
            'initial_frame_count': ParameterValue(initial_frame_count, value_type=int),
            'alpha_translation': ParameterValue(alpha_translation, value_type=float),
            'alpha_rotation': ParameterValue(alpha_rotation, value_type=float),
            't21_x_mm': ParameterValue(t21_x_mm, value_type=float),
            't21_y_mm': ParameterValue(t21_y_mm, value_type=float),
            't21_z_mm': ParameterValue(t21_z_mm, value_type=float),
            't0a_x_mm': ParameterValue(t0a_x_mm, value_type=float),
            't0a_y_mm': ParameterValue(t0a_y_mm, value_type=float),
            't0a_z_mm': ParameterValue(t0a_z_mm, value_type=float),
        }]
    )
    
    # ========== franka_control_trans: show_desired_matrix ==========
    show_desired_matrix_node = Node(
        package='franka_control_trans',
        executable='show_desired_matrix',
        name='show_desired_matrix',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
    )

    delayed_pose_publisher = TimerAction(
        period=2.0,
        actions=[
            pose_publisher_node,
            # show_desired_matrix_node
            ]
    )
    
    # ========== RViz ==========
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=[
            '-d', rviz_config_path,
            '--ros-args', '--log-level', log_level
            ],
    )

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
        franka_node,
        delayed_pose_publisher,
        # rviz_node
    ])
