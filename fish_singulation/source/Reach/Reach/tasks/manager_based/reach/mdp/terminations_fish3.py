from __future__ import annotations
import torch
from isaaclab.envs import ManagerBasedRLEnv

_FISH_KEYS = ("fish", "fish2", "fish3")
_PARKED_Z = 5.0


def max_episode_length(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def all_fish_lost(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate when every non-delivered fish has fallen off the table.

    Fires when: at least one fish is below the table (z < 0.15) AND no fish
    remain active (0.15 < z < 4.0). Delivered fish (z ≈ 5.0) are excluded —
    a mix of delivered + fallen should not trigger this; it means partial success.
    """
    any_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    any_fallen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for k in _FISH_KEYS:
        z = env.scene[k].data.root_pos_w[:, 2]
        any_active = any_active | ((z > 0.15) & (z < _PARKED_Z - 1.0))
        any_fallen = any_fallen | (z < 0.15)
    return any_fallen & ~any_active


def all_fish_delivered(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate successfully when all 3 fish have been parked (z > 4.0)."""
    threshold = _PARKED_Z - 1.0
    delivered = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for k in _FISH_KEYS:
        delivered = delivered & (env.scene[k].data.root_pos_w[:, 2] > threshold)
    env.extras["is_success"] = delivered.float()
    return delivered
