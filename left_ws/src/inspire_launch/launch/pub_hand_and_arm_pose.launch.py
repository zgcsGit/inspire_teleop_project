import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def build_venv_env(venv_path: str):
    """
    Build env overlay for running nodes inside a python venv.
    Assumes python3.10 (Ubuntu 22.04 / ROS2 Humble).
    """
    env = {
        'PATH': os.environ.get('PATH', ''),
        'PYTHONPATH': os.environ.get('PYTHONPATH', ''),
        'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', ''),
    }
    env['PATH'] = os.path.join(venv_path, 'bin') + ':' + env['PATH']
    env['PYTHONPATH'] = (
        os.path.join(venv_path, 'lib/python3.10/site-packages')
        + ':' + env['PYTHONPATH']
    )
    return env


def generate_launch_description():
    # =========================================================
    # Launch arguments
    # =========================================================
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level for all nodes'
    )

    subject_arg = DeclareLaunchArgument(
        'subject',
        default_value='lefthand',
        description='Vicon subject name'
    )

    log_level = LaunchConfiguration('log_level')
    subject = LaunchConfiguration('subject')

    # =========================================================
    # Python venv for retargeting / finger_map
    # =========================================================
    finger_map_venv = os.path.expanduser(os.environ.get('FINGER_MAP_VENV', '~/finger_map_venv'))
    venv_env = build_venv_env(finger_map_venv)

    # =========================================================
    # 1) Vicon UDP receiver (with subject)
    # =========================================================
    vicon_node = Node(
        package='vicon_udp_receiver',
        executable='with_subject',
        name='vicon_with_subject',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
        parameters=[{
            'subject': subject,
        }]
    )

    # =========================================================
    # 2) Retargeting + finger_map (delayed, venv)
    # =========================================================
    pub_csv_node = Node(
        package='inspire_retargeting',
        executable='pub_csv',
        name='pub_csv',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
        additional_env=venv_env
    )

    finger_mapper_node = Node(
        package='finger_map',
        executable='finger_mapper_node',
        name='finger_mapper',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
        additional_env=venv_env
    )

    delayed_retargeting = TimerAction(
        period=3.0,
        actions=[pub_csv_node, finger_mapper_node]
    )

    return LaunchDescription([
        log_level_arg,
        subject_arg,

        # Vicon first
        vicon_node,

        # Retargeting later
        delayed_retargeting,
    ])
