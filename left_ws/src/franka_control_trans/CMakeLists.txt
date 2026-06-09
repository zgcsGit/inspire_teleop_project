cmake_minimum_required(VERSION 3.5)
project(franka_control_trans)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(custom_msgs REQUIRED)
find_package(inspire_interfaces REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2_ros REQUIRED)



add_executable(pose_publisher src/pose_publisher.cpp)
ament_target_dependencies(
  pose_publisher 
  rclcpp 
  custom_msgs 
  inspire_interfaces
  tf2_ros
  ) 

target_link_libraries(pose_publisher Eigen3::Eigen)

add_executable(show_desired_matrix src/show_desired_matrix.cpp)
ament_target_dependencies(
  show_desired_matrix
  rclcpp
  tf2_ros
  custom_msgs
)
target_link_libraries(show_desired_matrix Eigen3::Eigen)

install(TARGETS
  pose_publisher
  show_desired_matrix
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)


ament_package()
