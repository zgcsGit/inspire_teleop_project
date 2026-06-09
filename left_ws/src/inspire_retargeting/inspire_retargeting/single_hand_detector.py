import numpy as np

# MANO坐标系转换矩阵
OPERATOR2MANO_RIGHT = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0]
])

OPERATOR2MANO_LEFT = np.array([
    [0, 0, -1],
    [1, 0, 0],
    [0, -1, 0]
])

class SingleHandDetector:
    def __init__(self, hand_type="Right", selfie=False):
        """
        hand_type: "Right" 或 "Left"
        selfie: 是否镜像（前置摄像头情况），用于决定左右手
        """
        self.operator2mano = OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        self.detected_hand_type = hand_type if selfie else inverse_hand_dict[hand_type]

    def detect_from_array(self, keypoint_3d_array):
        """
        输入：
            keypoint_3d_array: np.ndarray, shape (21,3)
        输出：
            num_box: int, 手数(固定为1)
            joint_pos: np.ndarray, shape (21,3), 转换到 MANO 坐标系
            keypoint_2d: None
            wrist_rot: np.ndarray, shape (3,3), 手腕旋转矩阵
        """
        if keypoint_3d_array.shape != (21, 3):
            raise ValueError("keypoint_3d_array 必须是 (21,3) 的 numpy 数组")

        # 将手腕置为原点
        keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]

        # 计算手腕坐标系
        wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)

        # 转换到MANO坐标系
        joint_pos = keypoint_3d_array @ wrist_rot @ self.operator2mano

        return 1, joint_pos, None, wrist_rot

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array):
        """
        根据手关键点计算手腕坐标系（只计算旋转）
        :param keypoint_3d_array: (21,3)
        :return: wrist_rot (3,3)
        """
        # points = keypoint_3d_array[[0, 5, 9], :]  # wrist, index_mcp, middle_mcp
        points = keypoint_3d_array[[0, 5, 9, 13, 17], :]  # wrist, index_mcp, middle_mcp, ring_mcp, pinky_mcp
        x_vector = points[0] - points[2]

        # SVD 拟合掌面法向量
        points_centered = points - np.mean(points, axis=0, keepdims=True)
        _, _, v = np.linalg.svd(points_centered)
        normal = v[2, :]

        # Gram–Schmidt 正交化
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1

        frame = np.stack([x, normal, z], axis=1)
        return frame
