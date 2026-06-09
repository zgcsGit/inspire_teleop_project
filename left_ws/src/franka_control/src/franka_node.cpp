#include "franka_control/franka_node.h"

FrankaNode::FrankaNode(std::string node_name, std::string robot_IP)
: Node(node_name), robot_(robot_IP) {
    callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    sub_options_.callback_group = callback_group_;

    rclcpp::QoS qos_profile(10);

    rclcpp::QoS pose_qos(1);
    pose_qos.best_effort();
    pose_qos.keep_last(1);

    desired_pose_subscription_ =
        this->create_subscription<custom_msgs::msg::DesiredPose>(
            "/left/desired_pose_matrix",
            qos_profile,
            std::bind(&FrankaNode::desiredPoseCallback, this, std::placeholders::_1),
            sub_options_
        );

    actual_ee_pose_matrix_publisher_ =
        this->create_publisher<custom_msgs::msg::DesiredPose>(
            "/frankaLeft/ee_pose_matrix",
            pose_qos
        );

    actual_ee_pose_timer_ =
        this->create_wall_timer(
            std::chrono::milliseconds(20),
            std::bind(&FrankaNode::publishActualEePoseMatrix, this)
        );

    control_shutdown = [&]() {
        robot_.stop();
        std::cout << "Stopped LEFT Franka Control" << std::endl;
    };

    rclcpp::on_shutdown(control_shutdown);

    control_thread_ = std::thread([this]() {
        try {
            std::array<double, 7> q_goal = {{
                0.0267305,
                0.0354752,
                -0.144137,
                -2.01915,
                -0.0240028,
                1.48962,
                0.919926
            }};

            MotionGenerator motion_generator(0.1, q_goal);

            std::cout << "MOVING LEFT ROBOT TO HOME POSITION." << std::endl;
            // std::cout << "robot states:" << robot_.readOnce() << std::endl;
            robot_.control(motion_generator);

            franka::Model model = robot_.loadModel();

            const double translational_stiffness{600.0};
            const double rotational_stiffness{30.0};

            Eigen::MatrixXd stiffness(6, 6);
            Eigen::MatrixXd damping(6, 6);

            stiffness.setZero();
            damping.setZero();

            stiffness.topLeftCorner(3, 3) =
                translational_stiffness * Eigen::MatrixXd::Identity(3, 3);

            stiffness.bottomRightCorner(3, 3) =
                rotational_stiffness * Eigen::MatrixXd::Identity(3, 3);

            damping.topLeftCorner(3, 3) =
                2.0 * std::sqrt(translational_stiffness) * Eigen::MatrixXd::Identity(3, 3);

            damping.bottomRightCorner(3, 3) =
                2.0 * std::sqrt(rotational_stiffness) * Eigen::MatrixXd::Identity(3, 3);

            franka::RobotState initial_state = robot_.readOnce();

            Eigen::Affine3d initial_transform(
                Eigen::Matrix4d::Map(initial_state.O_T_EE.data())
            );

            {
                std::lock_guard<std::mutex> lock(mutex_pose_);
                desired_pose_matrix_ = initial_transform.matrix();
            }

            position_d = initial_transform.translation();
            orientation_d = Eigen::Quaterniond(initial_transform.rotation());

            constexpr double MAX_DELTA_POS = 0.0001;
            constexpr double MAX_DELTA_ROT = 0.0005;

            impedance_control_callback =
                [&, this](
                    const franka::RobotState& robot_state,
                    franka::Duration /*duration*/
                ) -> franka::Torques {
                    std::array<double, 7> coriolis_array = model.coriolis(robot_state);
                    std::array<double, 42> jacobian_array =
                        model.zeroJacobian(franka::Frame::kEndEffector, robot_state);

                    Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(
                        coriolis_array.data()
                    );

                    Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(
                        jacobian_array.data()
                    );

                    Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(
                        robot_state.dq.data()
                    );

                    Eigen::Affine3d transform(
                        Eigen::Matrix4d::Map(robot_state.O_T_EE.data())
                    );

                    Eigen::Vector3d position = transform.translation();
                    Eigen::Quaterniond orientation(transform.rotation());

                    {
                        std::lock_guard<std::mutex> lock(mutex_actual_ee_pose_);
                        latest_actual_ee_pose_matrix_ = robot_state.O_T_EE;
                        has_actual_ee_pose_ = true;
                    }

                    Eigen::Vector3d position_target;
                    Eigen::Quaterniond orientation_target;

                    {
                        std::lock_guard<std::mutex> lock(mutex_pose_);
                        position_target = desired_pose_matrix_.block<3, 1>(0, 3);
                        orientation_target =
                            Eigen::Quaterniond(desired_pose_matrix_.block<3, 3>(0, 0));
                    }

                    Eigen::Vector3d delta_pos = position_target - position_d;

                    if (delta_pos.norm() > MAX_DELTA_POS) {
                        delta_pos = delta_pos.normalized() * MAX_DELTA_POS;
                    }

                    position_d += delta_pos;

                    Eigen::Quaterniond delta_q = orientation_d.inverse() * orientation_target;

                    if (
                        delta_q.coeffs().dot(
                            Eigen::Quaterniond::Identity().coeffs()
                        ) < 0.0
                    ) {
                        delta_q.coeffs() << -delta_q.coeffs();
                    }

                    Eigen::AngleAxisd delta_aa(delta_q);

                    if (delta_aa.angle() > MAX_DELTA_ROT) {
                        delta_aa.angle() = MAX_DELTA_ROT;
                        delta_q = Eigen::Quaterniond(delta_aa);
                    }

                    orientation_d = orientation_d * delta_q;
                    orientation_d.normalize();

                    Eigen::Matrix<double, 6, 1> error;
                    error.head(3) = position - position_d;

                    if (orientation_d.coeffs().dot(orientation.coeffs()) < 0.0) {
                        orientation.coeffs() << -orientation.coeffs();
                    }

                    Eigen::Quaterniond error_quaternion(
                        orientation.inverse() * orientation_d
                    );

                    error.tail(3) << error_quaternion.x(),
                                     error_quaternion.y(),
                                     error_quaternion.z();

                    error.tail(3) = -transform.rotation() * error.tail(3);

                    Eigen::VectorXd tau_task(7);
                    Eigen::VectorXd tau_d(7);

                    tau_task =
                        jacobian.transpose()
                        * (-stiffness * error - damping * (jacobian * dq));

                    tau_d = tau_task + coriolis;

                    std::array<double, 7> tau_d_array{};
                    Eigen::VectorXd::Map(tau_d_array.data(), 7) = tau_d;

                    return tau_d_array;
                };

            std::cout << "WARNING: Starting LEFT Cartesian impedance control."
                      << std::endl;

            while (rclcpp::ok()) {
                try {
                    robot_.control(impedance_control_callback);
                } catch (const franka::ControlException& e) {
                    std::cout << "Control exception: " << e.what() << std::endl;
                    std::cout << "Running error recovery..." << std::endl;
                    robot_.automaticErrorRecovery();
                }
            }

        } catch (const franka::Exception& e) {
            std::cout << e.what() << std::endl;

            control_shutdown();

            std::cout << "Shutting down ROS 2 process" << std::endl;
            rclcpp::shutdown();
        }
    });
}

FrankaNode::~FrankaNode() {
    if (control_thread_.joinable()) {
        std::cout << "Joining control thread" << std::endl;
        control_thread_.join();
    }

    std::cout << "Shutting down LEFT Franka control node" << std::endl;
}

void FrankaNode::desiredPoseCallback(
    const custom_msgs::msg::DesiredPose::SharedPtr msg
) {
    std::lock_guard<std::mutex> lock(mutex_pose_);

    Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::ColMajor>> mat(
        msg->pose_matrix.data()
    );

    Eigen::Matrix4d T_diff_inv;

    T_diff_inv <<
        0.990186, -0.040235,  0.133679, -0.016448,
        0.028145,  0.995958,  0.085489,  0.004888,
    -0.136962, -0.080571,  0.987314,  0.002884,
        0.0,       0.0,       0.0,       1.0;

    desired_pose_matrix_ = mat * T_diff_inv;

    double x = desired_pose_matrix_(0, 3);
    double y = desired_pose_matrix_(1, 3);
    double z = desired_pose_matrix_(2, 3);

    x = std::clamp(x, X_RANGE[0], X_RANGE[1]);
    y = std::clamp(y, Y_RANGE[0], Y_RANGE[1]);
    z = std::clamp(z, Z_RANGE[0], Z_RANGE[1]);

    desired_pose_matrix_(0, 3) = x;
    desired_pose_matrix_(1, 3) = y;
    desired_pose_matrix_(2, 3) = z;
}

void FrankaNode::publishActualEePoseMatrix() {
    std::lock_guard<std::mutex> lock(mutex_actual_ee_pose_);

    if (!has_actual_ee_pose_) {
        return;
    }

    actual_ee_pose_matrix_msg_.pose_matrix = latest_actual_ee_pose_matrix_;
    actual_ee_pose_matrix_publisher_->publish(actual_ee_pose_matrix_msg_);
}
