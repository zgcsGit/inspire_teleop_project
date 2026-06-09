#include "rclcpp/rclcpp.hpp"
#include "franka_control/franka_node.h"

namespace
{
std::string getRobotIpFromParams()
{
    constexpr const char* default_robot_ip = "12.1.1.5";

    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(true);

    auto param_node = std::make_shared<rclcpp::Node>("franka_param_loader", options);
    if (!param_node->has_parameter("robot_ip")) {
        param_node->declare_parameter<std::string>("robot_ip", default_robot_ip);
    }

    return param_node->get_parameter("robot_ip").as_string();
}
}  // namespace

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    std::string robot_IP = getRobotIpFromParams();

    std::shared_ptr<FrankaNode> robot =
        std::make_shared<FrankaNode>("franka_left", robot_IP);

    rclcpp::executors::MultiThreadedExecutor exec;
    exec.add_node(robot);
    exec.spin();

    std::cout << "Shutting down ROS 2 process" << std::endl;
    rclcpp::shutdown();

    return 0;
}
