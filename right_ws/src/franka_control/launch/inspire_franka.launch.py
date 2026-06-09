from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # ========== 声明可选参数 ==========
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level for both nodes'
    )

    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='12.1.1.6',
        description='IP address of the Franka robot'
    )
    desired_pose_topic_arg = DeclareLaunchArgument(
        'desired_pose_topic',
        default_value='/right/desired_pose_matrix',
        description='Desired pose topic consumed by the right Franka controller'
    )
    actual_ee_pose_topic_arg = DeclareLaunchArgument(
        'actual_ee_pose_topic',
        default_value='/frankaRight/ee_pose_matrix',
        description='Actual end-effector pose topic published by the right Franka controller'
    )
    input_hand_topic_arg = DeclareLaunchArgument(
        'input_hand_topic',
        default_value='/Righthandpoint',
        description='Right hand keypoint topic used to generate desired poses'
    )

    # Launch 参数对象
    log_level = LaunchConfiguration('log_level')
    robot_ip = LaunchConfiguration('robot_ip')
    desired_pose_topic = LaunchConfiguration('desired_pose_topic')
    actual_ee_pose_topic = LaunchConfiguration('actual_ee_pose_topic')
    input_hand_topic = LaunchConfiguration('input_hand_topic')

    # ========== 定义第一个节点 ==========
    franka_node = Node(
        package='franka_control',
        executable='franka_node',
        name='franka',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
        parameters=[{
            'robot_ip': robot_ip,
            'desired_pose_topic': desired_pose_topic,
            'actual_ee_pose_topic': actual_ee_pose_topic,
        }]
    )

    # ========== 定义第二个节点 ==========
    pose_publisher_node = Node(
        package='franka_control_trans',
        executable='pose_publisher',
        name='pose_publisher',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
        parameters=[{
            'input_hand_topic': input_hand_topic,
            'desired_pose_topic': desired_pose_topic,
        }]
    )
    
    # ========== franka_control_trans: show_desired_matrix ==========
    show_desired_matrix_node = Node(
        package='franka_control_trans',
        executable='show_desired_matrix',
        name='show_desired_matrix',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
        parameters=[{
            'desired_pose_topic': desired_pose_topic,
        }]
    )

    # ========== 延迟启动第二个节点 ==========
    delayed_pose_publisher = TimerAction(
        period=3.0,  # 延迟3秒启动
        actions=[
            pose_publisher_node,
            show_desired_matrix_node
            ]
    )
    
    # ========== 返回 LaunchDescription ==========
    return LaunchDescription([
        log_level_arg,
        robot_ip_arg,
        desired_pose_topic_arg,
        actual_ee_pose_topic_arg,
        input_hand_topic_arg,
        franka_node,
        delayed_pose_publisher,
        # rviz_node
    ])
