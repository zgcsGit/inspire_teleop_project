# finger_map/finger_model.py
import numpy as np

# ============================================================
# 共同工具：把 (x->y) 的 forward 模型变成 (y->x) 的反查函数
# 约束：角度越大 -> 控制值越小（即 y 随 x 增大而减小）
# ============================================================

def _build_inverse_lut_from_forward(poly_forward, x_min=0.0, x_max=1000.0, step=1.0):
    """
    poly_forward: function(x)->y
    returns: (ys_inc, xs_inc, y_min, y_max) for np.interp
    """
    xs = np.arange(x_min, x_max + step, step, dtype=np.float32)
    ys = np.array([poly_forward(float(x)) for x in xs], dtype=np.float32)

    # np.interp要求自变量递增，所以我们准备 y 递增的版本
    # 如果 y 随 x 增大递减，那么 ys[0] > ys[-1]，需要翻转
    if ys[0] > ys[-1]:
        ys_inc = ys[::-1]
        xs_inc = xs[::-1]
    else:
        ys_inc = ys
        xs_inc = xs

    y_min = float(ys_inc[0])
    y_max = float(ys_inc[-1])
    return ys_inc, xs_inc, y_min, y_max


# ============================================================
# 四指（index/middle/ring/pinky）官方多项式：y(x)
# y = a x^3 + b x^2 + c x + d
# x: 0..1000  (control)
# y: rad      (joint angle)
# 期望：角度越大 -> 控制越小（反向）
# ============================================================

A_F = -5.131897e-10
B_F =  9.124559e-7
C_F = -1.820094e-3
D_F =  1.419092323

def _poly_finger_angle(x: float) -> float:
    return ((A_F * x + B_F) * x + C_F) * x + D_F

_YS_F_INC, _XS_F_INC, _F_ANGLE_MIN, _F_ANGLE_MAX = _build_inverse_lut_from_forward(
    _poly_finger_angle, 0.0, 1000.0, 1.0
)

def map_finger(angle_rad: float) -> float:
    """
    四指：输入 proximal_joint 角度(rad) -> 输出控制量(0..1000)
    单调保证：angle越大 -> x越小（因为 forward y 随 x 递减）
    """
    a = float(np.clip(angle_rad, _F_ANGLE_MIN, _F_ANGLE_MAX))
    return float(np.interp(a, _YS_F_INC, _XS_F_INC))


# ============================================================
# 拇指 pitch 官方多项式：y(x)
# y = a x^3 + b x^2 + c x + d
# ============================================================

A_TP =  8.12114e-11
B_TP = -3.518841e-8
C_TP = -6.32529e-4
D_TP =  0.5869264679

def _poly_thumb_pitch_angle(x: float) -> float:
    return ((A_TP * x + B_TP) * x + C_TP) * x + D_TP

_YS_TP_INC, _XS_TP_INC, _TP_ANGLE_MIN, _TP_ANGLE_MAX = _build_inverse_lut_from_forward(
    _poly_thumb_pitch_angle, 0.0, 1000.0, 1.0
)

def map_thumb_pitch(angle_rad: float) -> float:
    """
    thumb pitch：输入 thumb_proximal_pitch_joint 角度(rad) -> 输出控制量(0..1000)
    单调保证：angle越大 -> x越小
    """
    a = float(np.clip(angle_rad, _TP_ANGLE_MIN, _TP_ANGLE_MAX))
    return float(np.interp(a, _YS_TP_INC, _XS_TP_INC))


# ============================================================
# 拇指 yaw 官方线性公式：y = m x + b
# 给的是：y = -0.001164127... * x + 1.1641485627
# => x = (y - b)/m
# 由于 m<0，所以 angle越大 -> x越小（符合你的“反比例”要求）
# ============================================================

_YAW_M = -0.00116412704172577
_YAW_B =  1.16414856276526

def map_thumb_yaw(angle_rad: float) -> float:
    """
    thumb yaw：输入 thumb_proximal_yaw_joint 角度(rad) -> 输出控制量(0..1000)
    """
    x = (float(angle_rad) - _YAW_B) / _YAW_M  # m<0 => angle越大 -> x越小
    return float(np.clip(x, 0.0, 1000.0))
