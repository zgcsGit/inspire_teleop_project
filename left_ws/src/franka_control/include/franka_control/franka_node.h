#pragma once

#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/robot.h>
#include <franka/model.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "custom_msgs/msg/desired_pose.hpp"

#include "franka_control/motion_generator.h"


class FrankaNode : public rclcpp::Node {
public:
    FrankaNode(std::string node_name, std::string robot_IP);
    ~FrankaNode();

private:
    franka::Robot robot_;

    std::thread control_thread_;

    rclcpp::SubscriptionOptions sub_options_;
    rclcpp::CallbackGroup::SharedPtr callback_group_;

    rclcpp::Subscription<custom_msgs::msg::DesiredPose>::SharedPtr desired_pose_subscription_;
    Eigen::Matrix4d desired_pose_matrix_;
    std::mutex mutex_pose_;

    rclcpp::Publisher<custom_msgs::msg::DesiredPose>::SharedPtr actual_ee_pose_matrix_publisher_;
    custom_msgs::msg::DesiredPose actual_ee_pose_matrix_msg_;
    rclcpp::TimerBase::SharedPtr actual_ee_pose_timer_;

    std::mutex mutex_actual_ee_pose_;
    std::array<double, 16> latest_actual_ee_pose_matrix_{};
    bool has_actual_ee_pose_ = false;

    Eigen::Vector3d position_d;
    Eigen::Quaterniond orientation_d;

    const double X_RANGE[2] = {0.0509428, 0.663813};
    const double Y_RANGE[2] = {-0.496687, 0.663813};
    const double Z_RANGE[2] = {-0.0290063, 1.649137};

    void desiredPoseCallback(const custom_msgs::msg::DesiredPose::SharedPtr msg);
    void publishActualEePoseMatrix();

    std::function<franka::Torques(
        const franka::RobotState&,
        franka::Duration
    )> impedance_control_callback;

    std::function<void()> control_shutdown;
};