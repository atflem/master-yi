from __future__ import annotations
import torch
from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Arm joints normalised to [-1, 1] via hard limits. Shape: (N, 6)"""
    robot = env.scene["robot"]
    q = robot.data.joint_pos[:, :6]
    lims = robot.data.joint_pos_limits[:, :6]
    rng = (lims[..., 1] - lims[..., 0]).clamp(min=0.1)
    return 2.0 * (q - lims[..., 0]) / rng - 1.0


def joint_vel_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Arm joint velocities scaled to [-1, 1]. Shape: (N, 6)"""
    return (env.scene["robot"].data.joint_vel[:, :6] / 6.283).clamp(-1.0, 1.0)


def ee_pos_rel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """EE position relative to robot base. Shape: (N, 3)"""
    ee = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    base = env.scene["robot"].data.root_pos_w
    return ee - base


def gripper_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gripper openness [0=open, 1=closed]. Shape: (N, 1)"""
    robot = env.scene["robot"]
    idx = robot.find_joints("finger_joint")[0][0]
    return (robot.data.joint_pos[:, idx] / 0.695).clamp(0.0, 1.0).unsqueeze(-1)


def fish_pos_rel(env: ManagerBasedRLEnv, fish_key: str = "fish") -> torch.Tensor:
    """Fish position relative to robot base. Shape: (N, 3)"""
    return env.scene[fish_key].data.root_pos_w - env.scene["robot"].data.root_pos_w


def fish_yaw_obs(env: ManagerBasedRLEnv, fish_key: str = "fish") -> torch.Tensor:
    """Fish yaw angle in [-pi, pi]. Shape: (N, 1)"""
    q = env.scene[fish_key].data.root_quat_w
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw.unsqueeze(-1)


def prev_action_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Previous action [dx, dy, dz, gripper]. Shape: (N, 4)"""
    return env.action_manager.prev_action
