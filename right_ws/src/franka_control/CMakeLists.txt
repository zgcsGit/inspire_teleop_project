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
find_package(custom_msgs REQUIRED)

find_package(Eigen3 REQUIRED)
if(DEFINED LIBFRANKA_BUILD_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${LIBFRANKA_BUILD_DIR}")
elseif(DEFINED ENV{LIBFRANKA_BUILD_DIR})
  list(APPEND CMAKE_PREFIX_PATH "$ENV{LIBFRANKA_BUILD_DIR}")
endif()
find_package(Franka REQUIRED)
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
  custom_msgs
)

ament_package()
