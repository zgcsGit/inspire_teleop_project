#include "rclcpp/rclcpp.hpp"
#include "custom_msgs/msg/desired_pose.hpp"
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <Eigen/Dense>
#include <iomanip>   // for std::setw and std::setprecision
#include <sstream>   // for std::ostringstream

class ShowDesiredMatrixNode : public rclcpp::Node
{
public:
    ShowDesiredMatrixNode()
    : Node("show_desired_matrix")
    {
        // --- 发布静态 world 坐标系 ---
        // 在构造函数里
        static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
        broadcast_world_origin();


        // --- 订阅 desired_pose_matrix ---
        subscription_ = this->create_subscription<custom_msgs::msg::DesiredPose>(
            "/desired_pose_matrix", 10,
            std::bind(&ShowDesiredMatrixNode::pose_callback, this, std::placeholders::_1)
        );

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        RCLCPP_INFO(this->get_logger(), "ShowDesiredMatrixNode initialized.");
    }

private:
    void broadcast_world_origin()
    {
        geometry_msgs::msg::TransformStamped world_msg;
        world_msg.header.stamp = this->now();
        world_msg.header.frame_id = "map";    // 可以是全局参考系
        world_msg.child_frame_id = "world";   // 我们固定的世界坐标系

        // 原点
        world_msg.transform.translation.x = 0.0;
        world_msg.transform.translation.y = 0.0;
        world_msg.transform.translation.z = 0.0;

        // 单位旋转（无旋转）
        world_msg.transform.rotation.x = 0.0;
        world_msg.transform.rotation.y = 0.0;
        world_msg.transform.rotation.z = 0.0;
        world_msg.transform.rotation.w = 1.0;

        static_broadcaster_->sendTransform(world_msg);

        RCLCPP_INFO(this->get_logger(), "Static world frame published at origin (0,0,0) with identity rotation.");
    }

    void pose_callback(const custom_msgs::msg::DesiredPose::SharedPtr msg)
    {
        if (msg->pose_matrix.size() != 16) {
            RCLCPP_WARN(this->get_logger(), "Invalid pose_matrix size: %zu", msg->pose_matrix.size());
            return;
        }

        // 将 std::array 转换为 Eigen::Matrix4d (列优先)
        Eigen::Matrix4d pose_matrix;
        for (int j = 0; j < 4; ++j)  // 列
            for (int i = 0; i < 4; ++i)  // 行
                pose_matrix(i, j) = msg->pose_matrix[j * 4 + i];

        //打印 pose_matrix
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4);
        oss << "\nReceived pose_matrix:\n";
        for (int i = 0; i < 4; ++i) {
            oss << "[ ";
            for (int j = 0; j < 4; ++j) {
                oss << std::setw(8) << pose_matrix(i, j) << " ";
            }
            oss << "]\n";
        }
        // RCLCPP_INFO(this->get_logger(), "%s", oss.str().c_str());

        // --- 提取 R 和 t ---
        Eigen::Matrix3d R = pose_matrix.topLeftCorner<3,3>();
        Eigen::Vector3d t = pose_matrix.topRightCorner<3,1>();

        // --- 构造 TF 消息 ---
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = this->now();
        tf_msg.header.frame_id = "world";        // 相对于固定 world
        tf_msg.child_frame_id = "desired_pose";

        tf_msg.transform.translation.x = t.x();
        tf_msg.transform.translation.y = t.y();
        tf_msg.transform.translation.z = t.z();

        Eigen::Quaterniond q(R);
        tf_msg.transform.rotation.x = q.x();
        tf_msg.transform.rotation.y = q.y();
        tf_msg.transform.rotation.z = q.z();
        tf_msg.transform.rotation.w = q.w();

        // --- 发布 TF ---
        tf_broadcaster_->sendTransform(tf_msg);

        // RCLCPP_INFO(this->get_logger(),
        //     "Broadcast TF: t=(%.3f, %.3f, %.3f)", t.x(), t.y(), t.z());
    }

    rclcpp::Subscription<custom_msgs::msg::DesiredPose>::SharedPtr subscription_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ShowDesiredMatrixNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
