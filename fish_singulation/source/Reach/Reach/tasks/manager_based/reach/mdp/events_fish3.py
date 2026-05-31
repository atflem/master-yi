from __future__ import annotations
import torch
from isaaclab.envs import ManagerBasedRLEnv

_FISH_KEYS = ("fish", "fish2", "fish3")
_OUTFEED_TARGET_LOCAL = [0.60, -0.30, 0.285]  # centre of outfeed table (_TABLE_Z + _FISH_HALF_H)
_PARKED_Z = 5.0


def apply_outfeed_conveyor(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    conveyor_speed: float = 0.15,
) -> None:
    """Set constant -Y velocity for fish on the outfeed zone, simulating a conveyor belt.

    Directly assigns velocity rather than applying force, bypassing static friction entirely.
    Only active fish that have crossed the infeed/outfeed boundary (robot-relative y < -0.02)
    are affected.
    """
    if len(env_ids) == 0:
        return

    base_pos = env.scene["robot"].data.root_pos_w[env_ids]

    for fish_key in _FISH_KEYS:
        fish = env.scene[fish_key]
        pos = fish.data.root_pos_w[env_ids]

        active = (pos[:, 2] > 0.15) & (pos[:, 2] < (_PARKED_Z - 1.0))
        fish_y_rel = pos[:, 1] - base_pos[:, 1]
        on_outfeed = active & (fish_y_rel < -0.02)

        if not on_outfeed.any():
            continue

        lin_vel = fish.data.root_lin_vel_w[env_ids].clone()
        ang_vel = fish.data.root_ang_vel_w[env_ids].clone()
        lin_vel[on_outfeed, 0] = 0.0
        lin_vel[on_outfeed, 1] = -conveyor_speed
        lin_vel[on_outfeed, 2] = 0.0
        ang_vel[on_outfeed]    = 0.0
        fish.write_root_velocity_to_sim(
            torch.cat([lin_vel, ang_vel], dim=-1), env_ids=env_ids
        )


def remove_delivered_fish(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pos_threshold: float = 0.12,
    align_threshold: float = 0.75,
    y_delivered: float | None = None,
) -> None:
    """Teleport fish to z=5.0 once they reach the outfeed target.

    Default mode: distance from outfeed target center < pos_threshold AND alignment >= align_threshold.
    Play mode (y_delivered set): any fish with robot-relative Y < y_delivered counts as delivered,
    which covers the entire outfeed table regardless of where on it the fish lands.
    """
    if len(env_ids) == 0:
        return

    base_pos = env.scene["robot"].data.root_pos_w[env_ids]
    local = torch.tensor(_OUTFEED_TARGET_LOCAL, device=env.device, dtype=torch.float32)
    target_xy = (base_pos + local.unsqueeze(0))[:, :2]

    for fish_key in _FISH_KEYS:
        fish = env.scene[fish_key]
        pos = fish.data.root_pos_w[env_ids]

        active = (pos[:, 2] > 0.15) & (pos[:, 2] < (_PARKED_Z - 1.0))
        if not active.any():
            continue

        if y_delivered is not None:
            fish_y_rel = pos[:, 1] - base_pos[:, 1]
            should_park = active & (fish_y_rel < y_delivered)
        else:
            d_xy = torch.norm(pos[:, :2] - target_xy, dim=-1)
            q = fish.data.root_quat_w[env_ids]
            w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
            yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            alignment = (1.0 - torch.cos(2.0 * yaw)) / 2.0
            should_park = active & (d_xy < pos_threshold) & (alignment >= align_threshold)

        if not should_park.any():
            continue

        park_ids = env_ids[should_park]
        park_pos = fish.data.root_pos_w[park_ids].clone()
        park_pos[:, 2] = _PARKED_Z
        park_quat = fish.data.root_quat_w[park_ids]
        zero_vel = torch.zeros(len(park_ids), 6, device=env.device)
        state = torch.cat([park_pos, park_quat, zero_vel], dim=-1)
        fish.write_root_state_to_sim(state, env_ids=park_ids)
