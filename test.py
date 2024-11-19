import math
import numpy as np
from scipy.spatial.transform import Rotation as R

def quat_2_euler(quaternion):
    """
    Converts quaternion into euler angles(format in xyz).

    :param quaternion: 1*4 quaternion
    :return: 1*3 euler angles
    """
    w, x, y, z = quaternion
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))

    input_value = 2 * (w * y - z * x)
    input_value = np.clip(input_value, -1, 1)  # 将输入值限制在 [-1, 1] 范围内
    pitch = np.arcsin(input_value)

    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    euler = np.array([roll, pitch, yaw])
    return euler

def euler_2_quat(rpy, degrees: bool = False):
    """
    degrees:bool
    True is the angle value, and False is the radian value
    """
    roll = rpy[0]
    pitch = rpy[1]
    yaw = rpy[2]
    if degrees:
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    return np.array([x, y, z, w])


# Define the quaternion
quat = [0.5, -0.5, -0.5, -0.5]
euler = [-1.57, 0, -1.57]
print(euler_2_quat(euler))
print(R.from_euler('xyz', euler).as_quat())
# euler_angles = quat_2_euler(quat)
# print("Euler Angles (roll, pitch, yaw):", euler_angles)