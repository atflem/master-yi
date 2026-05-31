from __future__ import annotations
import torch
from isaaclab.envs import ManagerBasedRLEnv

_FISH_KEYS = ("fish", "fish2", "fish3")
_TABLE_Z = 0.225
_FISH_HALF_H = 0.06
_OUTFEED_TARGET_LOCAL = [0.60, -0.30, _TABLE_Z + _FISH_HALF_H]
_PARKED_Z = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["ee_frame"].data.target_pos_w[..., 0, :]


def _base_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_pos_w


def _outfeed_target_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    local = torch.tensor(_OUTFEED_TARGET_LOCAL, device=env.device, dtype=torch.float32)
    return _base_pos(env) + local.unsqueeze(0)


def _fish_yaw(env: ManagerBasedRLEnv, fish_key: str) -> torch.Tensor:
    q = env.scene[fish_key].data.root_quat_w
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _is_active(env: ManagerBasedRLEnv, fish_key: str) -> torch.Tensor:
    """True for fish that are on/above the table and not yet delivered (parked)."""
    z = env.scene[fish_key].data.root_pos_w[:, 2]
    return (z > 0.15) & (z < (_PARKED_Z - 1.0))


def _nearest_active_dist(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Distance from EE to nearest active fish. Returns 1e6 if no active fish."""
    ee = _ee_pos(env)
    min_d = torch.full((env.num_envs,), 1e6, device=env.device)
    for k in _FISH_KEYS:
        pos = env.scene[k].data.root_pos_w
        d = torch.norm(ee - pos, dim=-1)
        d = torch.where(_is_active(env, k), d, torch.full_like(d, 1e6))
        min_d = torch.minimum(min_d, d)
    return min_d


# ---------------------------------------------------------------------------
# Stage 1: approach
# ---------------------------------------------------------------------------

def approach_reward(env: ManagerBasedRLEnv, std: float = 0.35) -> torch.Tensor:
    """Dense Gaussian: EE to nearest active fish."""
    d = _nearest_active_dist(env)
    return torch.exp(-d ** 2 / (2.0 * std ** 2))


def reach_sparse(env: ManagerBasedRLEnv, threshold: float = 0.05) -> torch.Tensor:
    """Sparse +1 when EE is within threshold of any active fish."""
    return (_nearest_active_dist(env) < threshold).float()


# ---------------------------------------------------------------------------
# Stage 2: grasp + transport + delivery
# ---------------------------------------------------------------------------

def grasp_reward(env: ManagerBasedRLEnv, threshold: float = 0.06) -> torch.Tensor:
    """Closure reward when EE is within threshold of any active fish.

    Decoupled from proximity: closure is rewarded as a binary gate (near/not near)
    rather than a Gaussian product. This prevents the fish wiggling when gripped
    from collapsing the reward — the gate stays open as long as EE is within ~6 cm
    (just beyond the fish half-width of 4.5 cm).
    """
    d = _nearest_active_dist(env)
    near = (d < threshold).float()
    robot = env.scene["robot"]
    idx = robot.find_joints("finger_joint")[0][0]
    closure = (robot.data.joint_pos[:, idx] / 0.695).clamp(0.0, 1.0)
    return near * closure


def transport_reward(env: ManagerBasedRLEnv, std: float = 0.30) -> torch.Tensor:
    """Dense Gaussian: nearest active fish XY distance to outfeed target."""
    target_xy = _outfeed_target_w(env)[:, :2]
    min_d = torch.full((env.num_envs,), 1e6, device=env.device)
    for k in _FISH_KEYS:
        pos = env.scene[k].data.root_pos_w
        d = torch.norm(pos[:, :2] - target_xy, dim=-1)
        d = torch.where(_is_active(env, k), d, torch.full_like(d, 1e6))
        min_d = torch.minimum(min_d, d)
    return torch.exp(-min_d ** 2 / (2.0 * std ** 2))


def velocity_axis_alignment_reward(
    env: ManagerBasedRLEnv,
    min_speed: float = 0.02,
    max_speed: float = 0.30,
) -> torch.Tensor:
    """Reward fish for moving along their own major axis (head/tail-first).

    Computes |dot(fish_major_axis, fish_velocity_direction)| × speed:
      - 1.0 when fish moves exactly head/tail-first at max_speed
      - 0.0 when fish moves sideways, or is stationary (speed < min_speed)

    Combined with fish_velocity (toward outfeed in -Y), the only configuration
    that maximises both rewards simultaneously is yaw≈±90° moving toward outfeed —
    i.e. head-first transport. The robot is implicitly trained to rotate fish to
    the correct orientation during transport, not just at the destination.
    """
    reward = torch.zeros(env.num_envs, device=env.device)
    for k in _FISH_KEYS:
        vel = env.scene[k].data.root_lin_vel_w
        active = _is_active(env, k)
        yaw = _fish_yaw(env, k)
        major_axis = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)  # (N,2)
        vel_2d = vel[:, :2]
        speed = torch.norm(vel_2d, dim=-1)
        moving = speed > min_speed
        vel_dir = vel_2d / speed.clamp(min=1e-6).unsqueeze(-1)
        alignment = torch.abs((major_axis * vel_dir).sum(dim=-1))
        speed_scaled = speed.clamp(0.0, max_speed)
        term = torch.where(active & moving, speed_scaled * alignment, torch.zeros_like(speed))
        reward = reward + term
    return reward


def yaw_alignment_reward(env: ManagerBasedRLEnv, pos_threshold: float = 0.25) -> torch.Tensor:
    """Dense reward for fish near the outfeed being correctly oriented (yaw ≈ ±90°).

    Provides gradient for rotating fish into head-first position. Only fires when
    the fish is already close to the outfeed (within pos_threshold), so it doesn't
    interfere with transport. Metric: (1 - cos(2·yaw)) / 2 — peaks at 1 for yaw=±90°.
    """
    target_xy = _outfeed_target_w(env)[:, :2]
    reward = torch.zeros(env.num_envs, device=env.device)
    for k in _FISH_KEYS:
        pos = env.scene[k].data.root_pos_w
        active = _is_active(env, k)
        d_xy = torch.norm(pos[:, :2] - target_xy, dim=-1)
        near_outfeed = active & (d_xy < pos_threshold)
        yaw = _fish_yaw(env, k)
        alignment = (1.0 - torch.cos(2.0 * yaw)) / 2.0
        reward = reward + torch.where(near_outfeed, alignment, torch.zeros_like(alignment))
    return reward


def delivery_reward(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.12,
    align_threshold: float = 0.75,
) -> torch.Tensor:
    """Sparse +1 per active fish placed at outfeed target with head-first orientation.

    Head-first = long axis (X) aligned with outfeed direction (Y), i.e. yaw ≈ ±90°.
    Metric: (1 − cos(2·yaw)) / 2 == 1 at yaw=±90°, 0 at yaw=0°/180°.
    align_threshold=0.75 corresponds to yaw within ≈30° of ±90°.
    """
    target_xy = _outfeed_target_w(env)[:, :2]
    reward = torch.zeros(env.num_envs, device=env.device)
    for k in _FISH_KEYS:
        pos = env.scene[k].data.root_pos_w
        active = _is_active(env, k)
        d_xy = torch.norm(pos[:, :2] - target_xy, dim=-1)
        yaw = _fish_yaw(env, k)
        alignment = (1.0 - torch.cos(2.0 * yaw)) / 2.0
        delivered = active & (d_xy < pos_threshold) & (alignment >= align_threshold)
        reward = reward + delivered.float()
    return reward


def fish_velocity_reward(env: ManagerBasedRLEnv, scale: float = 1.0) -> torch.Tensor:
    """Potential-based progress reward: fish moving toward outfeed earns +, away earns -.

    Bidirectional clamp [-0.3, 0.3] makes this equivalent to a potential-based shaping
    term (reward ≈ d_prev − d_curr per step) without requiring stored state.  A stationary
    fish earns 0 — no hover attractor.  Moving away is penalised, encouraging the robot
    to maintain progress and avoid pushing fish backward.
    """
    target_xy = _outfeed_target_w(env)[:, :2]
    reward = torch.zeros(env.num_envs, device=env.device)
    for k in _FISH_KEYS:
        pos = env.scene[k].data.root_pos_w
        vel = env.scene[k].data.root_lin_vel_w
        active = _is_active(env, k)
        direction = target_xy - pos[:, :2]
        dist = torch.norm(direction, dim=-1, keepdim=True).clamp(min=1e-6)
        direction_norm = direction / dist
        vel_toward = (vel[:, :2] * direction_norm).sum(dim=-1).clamp(-0.3, 0.3)
        reward = reward + torch.where(active, vel_toward, torch.zeros_like(vel_toward))
    return reward * scale


def ee_separation_penalty(env: ManagerBasedRLEnv, threshold: float = 0.15) -> torch.Tensor:
    """Penalty when gripper is closed but EE is far from all active fish."""
    robot = env.scene["robot"]
    idx = robot.find_joints("finger_joint")[0][0]
    closure = (robot.data.joint_pos[:, idx] / 0.695).clamp(0.0, 1.0)
    d = _nearest_active_dist(env)
    return -(closure * (d > threshold).float())


def lift_reward(env: ManagerBasedRLEnv, lift_height: float = 0.10) -> torch.Tensor:
    """Reward for lifting active fish above resting height — gated on gripper closure.

    Returns closure × (height_above_rest / lift_height) per active fish.
    Gating on closure prevents arm-body collisions (open gripper sweeping into the fish)
    from satisfying this reward — the robot must actually close the gripper to earn it.
    """
    resting_z = _TABLE_Z + _FISH_HALF_H
    robot = env.scene["robot"]
    idx = robot.find_joints("finger_joint")[0][0]
    closure = (robot.data.joint_pos[:, idx] / 0.695).clamp(0.0, 1.0)
    reward = torch.zeros(env.num_envs, device=env.device)
    for k in _FISH_KEYS:
        pos = env.scene[k].data.root_pos_w
        active = _is_active(env, k)
        height_above_rest = (pos[:, 2] - resting_z).clamp(0.0, lift_height) / lift_height
        term = closure * height_above_rest
        reward = reward + torch.where(active, term, torch.zeros_like(term))
    return reward


# ---------------------------------------------------------------------------
# Stage 3: fall penalty
# ---------------------------------------------------------------------------

def fish_fall_penalty(env: ManagerBasedRLEnv, min_z: float = _TABLE_Z - 0.05) -> torch.Tensor:
    """Penalty for each non-parked fish that has fallen below the table surface."""
    penalty = torch.zeros(env.num_envs, device=env.device)
    for k in _FISH_KEYS:
        z = env.scene[k].data.root_pos_w[:, 2]
        parked = z > (_PARKED_Z - 1.0)
        fell = (~parked) & (z < min_z)
        penalty = penalty + fell.float()
    return -penalty


# ---------------------------------------------------------------------------
# Always: regularisation
# ---------------------------------------------------------------------------

def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return -torch.norm(
        env.action_manager.action - env.action_manager.prev_action, dim=-1
    )


def joint_limit_penalty(env: ManagerBasedRLEnv, margin: float = 0.05) -> torch.Tensor:
    robot = env.scene["robot"]
    q = robot.data.joint_pos[:, :6]
    lims = robot.data.joint_pos_limits[:, :6]
    below = (lims[..., 0] + margin - q).clamp(min=0.0)
    above = (q - (lims[..., 1] - margin)).clamp(min=0.0)
    return -(below + above).sum(dim=-1)
