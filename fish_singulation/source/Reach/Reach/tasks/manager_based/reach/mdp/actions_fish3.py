from __future__ import annotations
import math
import torch
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass


class JointLimitSafeIKAction(DifferentialInverseKinematicsAction):
    """DifferentialIK action with hard joint-limit clamping on the desired targets.

    After the IK solver computes desired joint positions, the targets are clamped
    to the robot's joint position limits (fallback: ±2π) before being passed to
    the actuators.  Without this clamp, the PD actuators apply runaway torques
    whenever the IK is asked for an infeasible target — a recurring catastrophic
    failure mode in long training runs.
    """

    def apply_actions(self) -> None:
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(
                ee_pos_curr, ee_quat_curr, jacobian, joint_pos
            )
        else:
            joint_pos_des = joint_pos.clone()

        # Clamp desired targets to the robot's joint position limits.
        # Falls back to ±2π for continuous/unlimited joints.
        lims = self._asset.data.joint_pos_limits[:, self._joint_ids]
        lo = lims[..., 0].clamp(min=-2.0 * math.pi)
        hi = lims[..., 1].clamp(max=2.0 * math.pi)
        joint_pos_des = joint_pos_des.clamp(lo, hi)

        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)


@configclass
class JointLimitSafeIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for :class:`JointLimitSafeIKAction`."""
    class_type: type = JointLimitSafeIKAction
