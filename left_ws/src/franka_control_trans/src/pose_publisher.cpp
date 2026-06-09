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
    const std::string input_topic =
        this->declare_parameter<std::string>("input_topic", "/Lefthandpoint");
    const std::string output_topic =
        this->declare_parameter<std::string>("output_topic", "/left/desired_pose_matrix");

    initial_frame_count_ = this->declare_parameter<int>("initial_frame_count", 21);
    alpha_ = this->declare_parameter<double>("alpha_translation", 0.3);
    alpha_r_ = this->declare_parameter<double>("alpha_rotation", 0.2);

    t21_x_mm_ = this->declare_parameter<double>("t21_x_mm", 66.0);
    t21_y_mm_ = this->declare_parameter<double>("t21_y_mm", 0.0);
    t21_z_mm_ = this->declare_parameter<double>("t21_z_mm", 66.0);

    t0a_x_mm_ = this->declare_parameter<double>("t0a_x_mm", 610.0 / 2.0 + 100.0);
    t0a_y_mm_ = this->declare_parameter<double>("t0a_y_mm", -441.0);
    t0a_z_mm_ = this->declare_parameter<double>("t0a_z_mm", -90.0);

    subscription_ = this->create_subscription<inspire_interfaces::msg::HandKeypoint>(
        input_topic, 10,
        std::bind(&HandToDesiredPoseNode::hand_callback, this, std::placeholders::_1));

    publisher_ = this->create_publisher<custom_msgs::msg::DesiredPose>(
        output_topic, 10);
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
      RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "Not enough keypoints! (%zu)", msg->points.size());
      return;
    }

    // HandKeypoint positions are already in meters.
    Eigen::Vector3d p0(msg->points[0].x, msg->points[0].y, msg->points[0].z);
    Eigen::Vector3d p9(msg->points[9].x, msg->points[9].y, msg->points[9].z);
    Eigen::Vector3d p13(msg->points[13].x, msg->points[13].y, msg->points[13].z);

    // Build the hand marker frame from wrist/palm landmarks.
    Eigen::Vector3d v1 = (p9 - p0);
    Eigen::Vector3d v2 = (p13 - p0);
    if (v1.norm() < 1e-6 || v2.norm() < 1e-6)
      return;

    Eigen::Vector3d z_axis = v1.normalized();
    Eigen::Vector3d temp_x = v2.normalized();

    Eigen::Vector3d x_axis = z_axis.cross(temp_x);
    if (x_axis.norm() < 1e-6)
      return;
    x_axis.normalize();

    Eigen::Vector3d y_axis = z_axis.cross(x_axis).normalized();
    x_axis = y_axis.cross(z_axis).normalized();

    // Measured hand marker pose.
    Eigen::Matrix3d R20_meas;
    R20_meas.col(0) = x_axis;
    R20_meas.col(1) = y_axis;
    R20_meas.col(2) = z_axis;

    Eigen::Vector3d t20_meas = p0;

    Eigen::Quaterniond q20_meas(R20_meas);
    q20_meas.normalize();

    // Wait for 21 frames before publishing smoothed poses.
    if (!initialized_)
    {
      t_buffer_.push_back(t20_meas);
      q_buffer_.push_back(q20_meas);

      if (t_buffer_.size() == static_cast<size_t>(initial_frame_count_))
      {
        t20_smooth_ = t_buffer_.back();
        q20_smooth_ = q_buffer_.back();
        initialized_ = true;
      }
      return;
    }

    // Smooth translation with EMA and rotation with SLERP.
    t20_smooth_ = alpha_ * t20_meas + (1.0 - alpha_) * t20_smooth_;
    q20_smooth_ = q20_smooth_.slerp(alpha_r_, q20_meas);
    q20_smooth_.normalize();

    Eigen::Matrix3d R20_smooth = q20_smooth_.toRotationMatrix();
    Eigen::Matrix4d T20 = make_T(R20_smooth, t20_smooth_);

    // Fixed left-hand mounting transform. Translation entries are in mm below.
    Eigen::Matrix4d T21;
    T21 << 0, 0, 1, t21_x_mm_,
           0, 1, 0, t21_y_mm_,
          -1, 0, 0, t21_z_mm_,
           0, 0, 0, 1;

    T21(0,3) /= 1000.0;
    T21(1,3) /= 1000.0;
    T21(2,3) /= 1000.0;

    Eigen::Matrix4d T12 = T21.inverse();

    // Fixed offset from the tracking origin to the left Franka base.
    // Translation entries are in mm below.
    Eigen::Matrix4d T0A;
    T0A << 1, 0, 0, t0a_x_mm_,
           0, 1, 0, t0a_y_mm_,
           0, 0, 1, t0a_z_mm_,
           0, 0, 0, 1;

    T0A(0,3) /= 1000.0;
    T0A(1,3) /= 1000.0;
    T0A(2,3) /= 1000.0;

    Eigen::Matrix4d T1A = T0A * T20 * T12;

    custom_msgs::msg::DesiredPose msg_out;
    msg_out.pose_matrix = {
        T1A(0,0), T1A(1,0), T1A(2,0), T1A(3,0),
        T1A(0,1), T1A(1,1), T1A(2,1), T1A(3,1),
        T1A(0,2), T1A(1,2), T1A(2,2), T1A(3,2),
        T1A(0,3), T1A(1,3), T1A(2,3), T1A(3,3)};

    publisher_->publish(msg_out);
  }

  std::deque<Eigen::Vector3d> t_buffer_;
  std::deque<Eigen::Quaterniond> q_buffer_;
  bool initialized_ = false;

  Eigen::Vector3d t20_smooth_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q20_smooth_ = Eigen::Quaterniond::Identity();

  double alpha_ = 0.3;
  double alpha_r_ = 0.2;

  int initial_frame_count_ = 21;
  double t21_x_mm_ = 66.0;
  double t21_y_mm_ = 0.0;
  double t21_z_mm_ = 66.0;
  double t0a_x_mm_ = 610.0 / 2.0 + 100.0;
  double t0a_y_mm_ = -441.0;
  double t0a_z_mm_ = -90.0;

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
