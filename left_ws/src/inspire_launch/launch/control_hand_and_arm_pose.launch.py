import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Previous local docker compose directory:
# /home/tp2/git/teleassembly-desdren/docker_launch_files
# Use launch arguments or environment variables for portable clones.
DEFAULT_COMPOSE_DIR = os.environ.get("FRANKA_DOCKER_COMPOSE_DIR", os.getcwd())
DEFAULT_SERVICE_NAME = os.environ.get("FRANKA_DOCKER_SERVICE", "franka_node_image")
# Previous container setup path: /docker_volume/ros2_ws/install/setup.bash
DEFAULT_CONTAINER_SETUP = os.environ.get("FRANKA_DOCKER_SETUP", "/docker_volume/ros2_ws/install/setup.bash")

def generate_launch_description():
    # ---------- args ----------
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level for all nodes'
    )
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='12.1.1.6',
        description='Robot IP (used only if container launch supports it)'
    )
    compose_dir_arg = DeclareLaunchArgument(
        'compose_dir',
        default_value=DEFAULT_COMPOSE_DIR,
        description='Directory that contains the Franka docker compose file'
    )
    docker_service_arg = DeclareLaunchArgument(
        'docker_service',
        default_value=DEFAULT_SERVICE_NAME,
        description='Docker compose service name for the Franka container'
    )

    log_level = LaunchConfiguration('log_level')
    robot_ip  = LaunchConfiguration('robot_ip')
    compose_dir = LaunchConfiguration('compose_dir')
    docker_service = LaunchConfiguration('docker_service')

    # ---------- (1) Robot arm: docker compose exec -> ros2 launch franka_control inspire_franka.launch.py ----------
    docker_launch_cmd = (
        "source /opt/ros/humble/setup.bash && "
        f"source {DEFAULT_CONTAINER_SETUP} && "
        "ros2 launch franka_control inspire_franka.launch.py "
        # 如果 inspire_franka.launch.py 支持参数，可加：
        # f"robot_ip:={robot_ip} log_level:={log_level}"
    )

    franka_docker_launch = ExecuteProcess(
        cmd=[
            "docker", "compose", "exec", "-T",
            docker_service,
            "bash", "-lc",
            docker_launch_cmd
        ],
        cwd=compose_dir,
        output="screen"
    )

    # 可选：稍微延迟一下启动 franka（比如让系统起来）
    franka_start = TimerAction(period=1.0, actions=[franka_docker_launch])

    # ---------- (2) Hand: inspire_hand_modbus ----------
    hand_modbus_node = Node(
        package='inspire_hand_modbus',
        executable='inspire_hand_modbus_topic',
        name='hand_modbus',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level]
    )

    # ---------- (3) Hand init topics (after hand_modbus is up) ----------
    set_force = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--once',
            '/set_force_data', 'inspire_interfaces/msg/SetForce1',
            '{finger_ids:[1,2,3,4,5,6],forces:[500,500,500,500,500,500]}'
        ],
        output='screen'
    )

    set_speed = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--once',
            '/set_speed_data', 'inspire_interfaces/msg/SetSpeed1',
            '{finger_ids:[1,2,3,4,5,6],speeds:[500,500,500,500,500,500]}'
        ],
        output='screen'
    )

    set_angle = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--once',
            '/set_angle_data', 'inspire_interfaces/msg/SetAngle1',
            '{finger_ids:[1,2,3,4,5,6],angles:[1000,1000,1000,1000,1000,1000]}'
        ],
        output='screen'
    )

    # ⭐ 关键：保证“机械手在机械臂之后”
    # 做法：机械手整体延迟启动（比如 franka 启动后再等 3s）
    hand_bringup = TimerAction(
        period=4.0,  # = 1.0(franka_start) + 3.0(缓冲)，你可按实际调整
        actions=[
            hand_modbus_node,
            # 再给 modbus 节点 1s 时间起来，再发初始化 topic
            TimerAction(period=1.0, actions=[set_force]),
            TimerAction(period=1.2, actions=[set_speed]),
            TimerAction(period=1.4, actions=[set_angle]),
        ]
    )

    return LaunchDescription([
        log_level_arg,
        robot_ip_arg,
        compose_dir_arg,
        docker_service_arg,
        franka_start,
        hand_bringup,
    ])
