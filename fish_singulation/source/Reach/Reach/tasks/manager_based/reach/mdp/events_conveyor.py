# Copyright (c) 2024, Your Name.
# SPDX-License-Identifier: BSD-3-Clause
"""
Conveyor belt event functions for UR10e fish singulation.

Conveyor implementation
-----------------------
Rather than using kinematic rigid bodies (which have GPU-pipeline instancing
issues and uncertain velocity semantics when driven post-physics), we keep the
outfeed tables as ordinary static AssetBaseCfg objects and directly override
each fish's root velocity once it crosses onto the outfeed area.

Calling this event every policy step (interval_range_s = (policy_dt, policy_dt))
means the belt velocity is re-imposed after every physics integration, so table
friction cannot slow the fish below the belt speed.  Fish that are still on the
infeed (y > threshold) are unaffected.

Usage in EventsCfg:
    outfeed_belt = EventTermCfg(
        func=_ev.apply_outfeed_belt_velocity,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={"belt_speed": 0.10, "threshold_y": -0.05},
    )
"""

from __future__ import annotations
import torch
from isaaclab.envs import ManagerBasedRLEnv


def apply_outfeed_belt_velocity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    belt_speed: float = 0.10,
    threshold_y: float = -0.05,
) -> None:
    """
    Simulate outfeed conveyor belts by directly setting fish velocity once they
    cross onto the outfeed (fish_y < robot_base_y + threshold_y).

    Only the Y component is overridden; X and Z are zeroed so the fish travels
    straight down its lane without lateral drift.

    Args:
        env_ids:     Environments to process (all envs each policy step).
        belt_speed:  Outfeed belt speed in m/s (positive value, direction is −Y).
        threshold_y: Y offset from robot base that marks the infeed/outfeed boundary.
    """
    if len(env_ids) == 0:
        return

    robot_base_y = env.scene["robot"].data.root_pos_w[env_ids, 1]
    thresh = robot_base_y + threshold_y   # world-frame Y threshold per env

    for fish_key in ("fish", "fish2"):
        fish = env.scene[fish_key]
        fish_y = fish.data.root_pos_w[env_ids, 1]
        on_outfeed = fish_y < thresh          # (N,) bool mask

        if not on_outfeed.any():
            continue

        # Indices within env_ids that are on the outfeed
        active_env_ids = env_ids[on_outfeed]

        # Build (N_active, 6) velocity: [vx=0, vy=−belt, vz=0, wx=0, wy=0, wz=0]
        vel = torch.zeros(len(active_env_ids), 6, device=env.device)
        vel[:, 1] = -belt_speed

        fish.write_root_velocity_to_sim(vel, env_ids=active_env_ids)
