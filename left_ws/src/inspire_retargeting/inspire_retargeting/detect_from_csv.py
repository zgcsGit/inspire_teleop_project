import pickle
from pathlib import Path
import os

import numpy as np
import tqdm
import tyro

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from inspire_retargeting.single_hand_detector import SingleHandDetector


def load_points_from_csv(file_path: str) -> np.ndarray:
    """
    从 CSV 加载每帧 21 个关键点的 3D 坐标
    CSV 格式: 每行 63 个值, x0,y0,z0,...,x20,y20,z20
    返回: shape (num_frames, 21, 3)
    """
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    num_frames = data.shape[0]
    points_all = data.reshape(num_frames, 21, 3)
    return points_all


def retarget_from_points(
    retargeting: SeqRetargeting, points_csv: str, output_path: str, hand_type: HandType
):
    points_all = load_points_from_csv(points_csv)
    data = []

    # 新建 SingleHandDetector，用于估计手腕坐标系
    hand_str = hand_type.name.lower()
    hand_detector = SingleHandDetector(hand_type=hand_str.capitalize(), selfie=False)

    # 先打印一次关节名，方便确认发布顺序
    joint_names = retargeting.optimizer.robot.dof_joint_names
    print(f"[Retargeting] dof = {len(joint_names)}")
    print(f"[Retargeting] joint_names = {joint_names}")

    with tqdm.tqdm(total=len(points_all)) as pbar:
        for idx, frame_pts in enumerate(points_all):
            # 使用 SingleHandDetector 直接处理 3D 点
            num_box, joint_pos, _, wrist_rot = hand_detector.detect_from_array(frame_pts)

            if num_box == 0 or joint_pos is None:
                pbar.update(1)
                continue

            # 构造 ref_value
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

            # 调用 retarget
            qpos = retargeting.retarget(ref_value)
            data.append(qpos)

            # —— 调试：仅打印前三帧的 qpos 信息 —— #
            if idx < 3:
                print(f"\n[Frame {idx}] qpos shape = {qpos.shape}")
                # 可按需减少打印精度
                np.set_printoptions(precision=4, suppress=True)
                print(f"[Frame {idx}] qpos = {qpos}")

            pbar.update(1)

    # 保存结果
    meta_data = dict(
        config_path="from_points_csv",
        dof=len(retargeting.optimizer.robot.dof_joint_names),
        joint_names=retargeting.optimizer.robot.dof_joint_names,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(dict(data=data, meta_data=meta_data), f)

    retargeting.verbose()
    print(f"处理完成，保存到 {output_path}")


def main(
    robot_name: RobotName,
    points_csv: str,
    output_path: str,
    retargeting_type: RetargetingType,
    hand_type: HandType,
):
    points_csv_path = Path(__file__).absolute().parent.parent / "data" / points_csv
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)

    robot_dir = Path(
        os.environ.get("DEX_RETARGETING_ROBOT_DIR", "")
        or "~/dex-retargeting/assets/robots/hands"
    ).expanduser()

    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    retarget_from_points(retargeting, points_csv_path, output_path, hand_type)


if __name__ == "__main__":
    tyro.cli(main)
