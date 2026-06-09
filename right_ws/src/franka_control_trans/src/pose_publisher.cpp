#include "rclcpp/rclcpp.hpp"
#include "custom_msgs/msg/desired_pose.hpp"
#include "inspire_interfaces/msg/hand_keypoint.hpp"
#include <Eigen/Dense>
#include <deque>

class HandToDesiredPoseNode : public rclcpp::Node
{
public:
  HandToDesiredPoseNode() : Node("hand_to_desired_pose")
  {
    const std::string input_hand_topic = this->declare_parameter<std::string>(
        "input_hand_topic", "/Righthandpoint");
    const std::string desired_pose_topic = this->declare_parameter<std::string>(
        "desired_pose_topic", "right/desired_pose_matrix");

    subscription_ = this->create_subscription<inspire_interfaces::msg::HandKeypoint>(
        input_hand_topic, 10,
        std::bind(&HandToDesiredPoseNode::hand_callback, this, std::placeholders::_1));

    publisher_ = this->create_publisher<custom_msgs::msg::DesiredPose>(
        desired_pose_topic, 10);

    RCLCPP_INFO(this->get_logger(), "HandToDesiredPoseNode initialized.");
  }

private:
  static Eigen::Matrix4d make_T(const Eigen::Matrix3d &R, const Eigen::Vector3d &t)
  {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.topLeftCorner<3, 3>() = R;
    T.topRightCorner<3, 1>() = t;
    return T;
  }

  void hand_callback(const inspire_interfaces::msg::HandKeypoint::SharedPtr msg)
  {
    if (msg->points.size() < 14)
    {
      RCLCPP_WARN(this->get_logger(), "Not enough keypoints! (%zu)", msg->points.size());
      return;
    }

    // --- 取关键点（单位：mm）---
    Eigen::Vector3d p0(msg->points[0].x, msg->points[0].y, msg->points[0].z);
    Eigen::Vector3d p9(msg->points[9].x, msg->points[9].y, msg->points[9].z);
    Eigen::Vector3d p13(msg->points[13].x, msg->points[13].y, msg->points[13].z);

    // --- 构造坐标系2（三轴在0中的方向），原点为p0 ---
    Eigen::Vector3d v1 = (p9 - p0);
    Eigen::Vector3d v2 = (p13 - p0);
    if (v1.norm() < 1e-6 || v2.norm() < 1e-6)
      return;

    Eigen::Vector3d z_axis = v1.normalized();
    Eigen::Vector3d temp_x = v2.normalized();

    Eigen::Vector3d x_axis = temp_x.cross(z_axis);
    if (x_axis.norm() < 1e-6)
      return;
    x_axis.normalize();

    Eigen::Vector3d y_axis = z_axis.cross(x_axis).normalized();
    x_axis = y_axis.cross(z_axis).normalized(); // 重新正交化

    // --- 绝对位姿：T20 = “2 在 0 中” ---
    Eigen::Matrix3d R20_meas;
    R20_meas.col(0) = x_axis;
    R20_meas.col(1) = y_axis;
    R20_meas.col(2) = z_axis;

    // Eigen::Vector3d t20_meas = p0 / 1000.0; // m
    Eigen::Vector3d t20_meas = p0 ; // m

    Eigen::Quaterniond q20_meas(R20_meas);
    q20_meas.normalize();

    // --- 收集初始21帧（用于初始化平滑状态）---
    if (!initialized_)
    {
      t_buffer_.push_back(t20_meas);
      q_buffer_.push_back(q20_meas);

      if (t_buffer_.size() == 21)
      {
        t20_smooth_ = t_buffer_.back();
        q20_smooth_ = q_buffer_.back();
        initialized_ = true;

        t_smooth_initialized_ = true;
        r_smooth_initialized_ = true;

        RCLCPP_INFO(this->get_logger(),
                    "Initialized smoothing with frame 21: t20=(%.3f, %.3f, %.3f)",
                    t20_smooth_.x(), t20_smooth_.y(), t20_smooth_.z());
      }
      else
      {
        RCLCPP_INFO(this->get_logger(),
                    "Collecting initial messages: %zu/21", t_buffer_.size());
      }
      return;
    }

    // --- 平滑绝对 T20 ---
    // 平移 EMA
    t20_smooth_ = alpha_ * t20_meas + (1.0 - alpha_) * t20_smooth_;

    // 旋转 SLERP
    q20_smooth_ = q20_smooth_.slerp(alpha_r_, q20_meas);
    q20_smooth_.normalize();

    Eigen::Matrix3d R20_smooth = q20_smooth_.toRotationMatrix();
    Eigen::Matrix4d T20 = make_T(R20_smooth, t20_smooth_); // “2 在 0 中”

    // // --- 固定已知：T21（“2 在 1 中”）--- 90度绕y轴，平移(66, 0, 66)mm 1 original
    Eigen::Matrix4d T21;
    T21 << 0, 0, 1, 66,
           0, 1, 0, 0,
          -1, 0, 0, 66,
           0, 0, 0, 1;

    // Eigen::Matrix4d T21; // 75度绕y轴，平移(45.24, 0, 56.95)mm  2 type
    // T21 << 0.258819, 0.0,  0.965926, 45.24,
    //       0.0,      1.0,  0.0,       0.0,
    //       -0.965926, 0.0,  0.258819, 56.95,
    //       0.0,      0.0,  0.0,       1.0;
      // 注意单位：如果 66 是 mm，且你其他都用 m，这里需要 /1000
    // 如果 66 已经是 m，就别除。通常这里应该是 mm -> m：
    T21(0,3) /= 1000.0;
    T21(1,3) /= 1000.0;
    T21(2,3) /= 1000.0;

    // 得到：T12 = “1 在 2 中”
    Eigen::Matrix4d T12 = T21.inverse();

    // --- 固定已知：T0A（“0 在 A 中”）---
    Eigen::Matrix4d T0A;
    T0A << 1, 0, 0, 610.0 / 2.0+100.0,
           0, 1, 0, 341.0,
           0, 0, 1, -170.0,
           0, 0, 0, 1;

    // 同样注意单位：如果这里是 mm，转换到 m
    T0A(0,3) /= 1000.0;
    T0A(1,3) /= 1000.0;
    T0A(2,3) /= 1000.0;

    // --- 按你的推导链条：T1A = T0A * T20 * T12 ---
    Eigen::Matrix4d T1A = T0A * T20 * T12;

    // --- 发布 T1A ---
    custom_msgs::msg::DesiredPose msg_out;
    msg_out.pose_matrix = {
        T1A(0,0), T1A(1,0), T1A(2,0), T1A(3,0),
        T1A(0,1), T1A(1,1), T1A(2,1), T1A(3,1),
        T1A(0,2), T1A(1,2), T1A(2,2), T1A(3,2),
        T1A(0,3), T1A(1,3), T1A(2,3), T1A(3,3)};

    publisher_->publish(msg_out);

    // RCLCPP_INFO(this->get_logger(),
    //             "Published T1A: t=(%.3f, %.3f, %.3f)",
    //             T1A(0,3), T1A(1,3), T1A(2,3));
  }

  // --- buffers / smooth state ---
  std::deque<Eigen::Vector3d> t_buffer_;
  std::deque<Eigen::Quaterniond> q_buffer_;
  bool initialized_ = false;

  Eigen::Vector3d t20_smooth_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q20_smooth_ = Eigen::Quaterniond::Identity();
  bool t_smooth_initialized_ = false;
  bool r_smooth_initialized_ = false;

  double alpha_ = 0.3;
  double alpha_r_ = 0.2;

  rclcpp::Subscription<inspire_interfaces::msg::HandKeypoint>::SharedPtr subscription_;
  rclcpp::Publisher<custom_msgs::msg::DesiredPose>::SharedPtr publisher_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HandToDesiredPoseNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
