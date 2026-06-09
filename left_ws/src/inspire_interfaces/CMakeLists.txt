cmake_minimum_required(VERSION 3.5)
project(inspire_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 生成接口
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GetForceAct1.msg"
  "msg/GetAngleAct1.msg"
  "msg/GetTouchAct1.msg"
  "msg/SetAngle1.msg"
  "msg/SetForce1.msg"
  "msg/SetSpeed1.msg"
  "msg/HandKeypoint.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_package()

