#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # -------------------------
    # Args
    # -------------------------
    rate_hz_arg = DeclareLaunchArgument(
        'rate_hz',
        default_value='15.0'
    )

    slop_sec_arg = DeclareLaunchArgument(
        'slop_sec',
        default_value='0.3'
    )
    out_dir_arg = DeclareLaunchArgument(
        'out_dir',
        default_value='~/inspire_teleop_data/right'
    )
    wrist_left_serial_arg = DeclareLaunchArgument(
        'wrist_left_serial',
        default_value="'213322073743'"
    )
    wrist_right_serial_arg = DeclareLaunchArgument(
        'wrist_right_serial',
        default_value="'828112071102'"
    )

    rate_hz = LaunchConfiguration('rate_hz')
    slop_sec = LaunchConfiguration('slop_sec')
    out_dir = LaunchConfiguration('out_dir')
    wrist_left_serial = LaunchConfiguration('wrist_left_serial')
    wrist_right_serial = LaunchConfiguration('wrist_right_serial')

    # -------------------------
    # Package paths
    # -------------------------
    realsense_launch_path = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch',
        'rs_launch.py'
    )

    kinect_launch_path = os.path.join(
        get_package_share_directory('azure_kinect_ros_driver'),
        'launch',
        'driver.launch.py'
    )

    record_launch_path = os.path.join(
        get_package_share_directory('multi_modal_data_collection'),
        'launch',
        'record_multimodal_with_timestamps.launch.py'
    )

    # -------------------------
    # 1) Left wrist RealSense
    # -------------------------
    camera_wrist_left = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch_path),
        launch_arguments={
            'camera_name': 'camera_wrist_left',
            'serial_no': wrist_left_serial,
            'enable_depth': 'false',
            'rgb_camera.profile': '960,540,30',
        }.items()
    )

    # -------------------------
    # 2) Right wrist RealSense
    # -------------------------
    camera_wrist_right = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch_path),
        launch_arguments={
            'camera_name': 'camera_wrist_right',
            'serial_no': wrist_right_serial,
            'enable_depth': 'false',
            'rgb_camera.profile': '960,540,30',
        }.items()
    )

    # -------------------------
    # 3) Azure Kinect
    # -------------------------
    azure_kinect = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(kinect_launch_path),
        launch_arguments={
            'depth_enabled': 'false',
            'color_enabled': 'true',
            'point_cloud': 'false',
            'rgb_point_cloud': 'false',
        }.items()
    )

    # -------------------------
    # 4) Record node/launch
    # -------------------------
    record_multimodal = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(record_launch_path),
        launch_arguments={
            'rate_hz': rate_hz,
            'slop_sec': slop_sec,
            'out_dir': out_dir,
        }.items()
    )

    # -------------------------
    # Sequential startup
    # -------------------------
    left_delayed = TimerAction(
        period=0.0,
        actions=[camera_wrist_left]
    )

    right_delayed = TimerAction(
        period=0.0,
        actions=[camera_wrist_right]
    )

    kinect_delayed = TimerAction(
        period=0.0,
        actions=[azure_kinect]
    )

    # 给三个相机一点启动和稳定时间
    record_delayed = TimerAction(
        period=3.0,
        actions=[record_multimodal]
    )

    return LaunchDescription([
        rate_hz_arg,
        slop_sec_arg,
        out_dir_arg,
        wrist_left_serial_arg,
        wrist_right_serial_arg,

        left_delayed,
        right_delayed,
        kinect_delayed,
        record_delayed,
    ])
