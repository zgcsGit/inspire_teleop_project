cmake_minimum_required(VERSION 3.5)
project(franka_control)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(custom_msgs REQUIRED)

find_package(Eigen3 REQUIRED)

# Previous local libfranka build path:
# /home/tp2/git/teleassembly-desdren/docker_volume/libfranka/build
# For a portable clone, either install libfranka system-wide or pass one of:
#   colcon build --cmake-args -DFranka_DIR=/path/to/libfranka/build
#   LIBFRANKA_BUILD_DIR=/path/to/libfranka/build colcon build
set(LIBFRANKA_BUILD_DIR "$ENV{LIBFRANKA_BUILD_DIR}" CACHE PATH "Optional libfranka build/install directory")
if(LIBFRANKA_BUILD_DIR)
  find_package(Franka REQUIRED HINTS "${LIBFRANKA_BUILD_DIR}")
else()
  find_package(Franka REQUIRED)
endif()

find_package(Threads REQUIRED)

add_executable(franka_node
  src/main.cpp
  src/franka_node.cpp
  include/franka_control/motion_generator.cpp
)

target_link_libraries(franka_node
  Franka::Franka
  Eigen3::Eigen
  Threads::Threads
)

target_include_directories(franka_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  include
)

ament_target_dependencies(
  franka_node
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  custom_msgs
)

install(TARGETS franka_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_dependencies(
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  custom_msgs
)

ament_package()