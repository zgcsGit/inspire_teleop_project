#include "rclcpp/rclcpp.hpp"
#include "franka_control/franka_node.h"
// #include "franka_control/control_right.hpp"


int main(int argc, char **argv) {

    rclcpp::init(argc, argv);

    // The launch file passes robot_ip as a ROS parameter; FrankaNode reads it
    // before opening the libfranka connection.
    std::shared_ptr<FrankaNode> robot = std::make_shared<FrankaNode>("franka_right");

    // Start subscriber callback
    rclcpp::executors::MultiThreadedExecutor exec;
    exec.add_node(robot);
    exec.spin();

    std::cout << "Shutting down ROS 2 process" << std::endl;
    rclcpp::shutdown();

    return 0;
}
