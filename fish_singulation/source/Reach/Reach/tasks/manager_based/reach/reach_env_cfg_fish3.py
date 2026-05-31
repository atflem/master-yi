# Copyright (c) 2024, Your Name.
# SPDX-License-Identifier: BSD-3-Clause
"""
Three-fish singulation with staged training curriculum.

Stages (simplest → full task):
  S1 — Approach   : learn to reach fish centroid.              UR10e-Fish3-S1-v0
  S2 — Transport  : pick and place fish to outfeed.            UR10e-Fish3-S2-v0
  S3 — Full task  : 3 fish + fall penalty.                     UR10e-Fish3-S3-v0

Transfer workflow:
  Train S1 → checkpoint → train S2 → checkpoint → train S3.
  All stages share identical 32-dim obs and 4-dim action space.

Scene (env-local, robot base at origin):
  In-feed  : x∈[0.20,1.00], y∈[0.00,0.90], surface z=0.225
  Out-feed  : x∈[0.20,1.00], y∈[-0.60,0.00], surface z=0.225
  Fish 1    : default (0.42, 0.15, 0.2485)
  Fish 2    : default (0.57, 0.30, 0.2485)
  Fish 3    : default (0.72, 0.45, 0.2485)
  Target    : (0.60, -0.30, 0.285) — centre of outfeed table

Head-first delivery: fish long axis (X) aligned with outfeed direction (Y), yaw≈±90°.
Delivered fish are teleported to z=5.0 and excluded from all future rewards.

Deformable fish: TODO — requires USD with PhysxDeformableBodyAPI + tet mesh.
Current proxy: soft-contact rigid bodies (compliant contact settings).
"""

from __future__ import annotations
import math
import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.managers import (
    ActionTerm,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .mdp import observations_fish3 as _o
from .mdp import rewards_fish3      as _r
from .mdp import terminations_fish3 as _t
from .mdp import events_fish3       as _ev
from .mdp.actions_fish3 import JointLimitSafeIKActionCfg
from .ur_gripper_new import UR10e_ROBOTIQ_CFG

_TABLE_Z     = 0.225
_PARKED_Z    = 5.0       # z height for delivered / permanently-parked fish
# Half-height of the fish mesh in metres — used to sit the fish on the table.
# If the fish floats above or clips through the table, adjust this value.
# If the USD origin is at the mesh bottom (not centre), set _FISH_HALF_H = 0.0.
_FISH_HALF_H = 0.06
_FISH_Z      = _TABLE_Z + _FISH_HALF_H
_POLICY_DT   = 0.02

# Physics-ready fish mesh (PhysicsRigidBodyAPI + ConvexHull collision added in Isaac Sim).
_FISH_USD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "fish_rigid.usd"))
# Scale: 0.01 if the USD was authored in centimetres (Isaac Sim default), 1.0 if metres.
# Do a quick visual run to verify — fish should be ~35 cm long on the table.
_FISH_USD_SCALE = (0.35, 0.35, 0.35)


# =============================================================================
# Scene
# =============================================================================

@configclass
class Fish3SceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8)),
    )

    robot: ArticulationCfg = UR10e_ROBOTIQ_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # Static in-feed table: x∈[0.20,1.00], y∈[0.00,0.90]
    infeed_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/InfeedTable",
        spawn=sim_utils.CuboidCfg(
            size=(0.80, 0.90, _TABLE_Z),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.20,
                dynamic_friction=0.15,
                restitution=0.02,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.35, 0.15), roughness=0.8,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.60, 0.45, _TABLE_Z / 2)),
    )

    # Static out-feed table: x∈[0.20,1.00], y∈[-0.90,0.00]
    outfeed_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/OutfeedTable",
        spawn=sim_utils.CuboidCfg(
            size=(0.80, 0.90, _TABLE_Z),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.20,
                dynamic_friction=0.15,
                restitution=0.02,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.30, 0.55, 0.30), roughness=0.6,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.60, -0.45, _TABLE_Z / 2)),
    )

    fish = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Fish",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_FISH_USD,
            scale=_FISH_USD_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=0.5,
                linear_damping=0.5,
                angular_damping=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0.001,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.40, 0.15, _FISH_Z), rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    fish2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Fish2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_FISH_USD,
            scale=_FISH_USD_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=0.5,
                linear_damping=0.5,
                angular_damping=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0.001,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.80, 0.15, _FISH_Z), rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    fish3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Fish3",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_FISH_USD,
            scale=_FISH_USD_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=0.5,
                linear_damping=0.5,
                angular_damping=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0.001,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.60, 0.45, _FISH_Z), rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link",
                name="end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.175)),
            )
        ],
    )

    num_envs: int = MISSING
    env_spacing: float = MISSING


# =============================================================================
# Actions: 3D Cartesian delta (arm) + continuous gripper
# =============================================================================

@configclass
class ActionsCfg:
    # JointLimitSafeIKActionCfg clamps IK targets to joint limits before commanding
    # the actuators — prevents runaway torques when the IK is given infeasible targets.
    arm_action: ActionTerm = JointLimitSafeIKActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ],
        body_name="wrist_3_link",
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.175)),
        scale=0.05,
        controller=DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=True,
            ik_method="dls",
        ),
    )

    # action=-1 → open (0 rad), action=+1 → closed (0.695 rad)
    # S3a disables this via self.actions.gripper_action = None (pushing only, no grasping).
    gripper_action: ActionTerm = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        scale=0.695,
        use_default_offset=True,
    )


# =============================================================================
# Observations: 32-dim, robot-relative positions
# =============================================================================

@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """
        32-dim flat observation (robot-relative).

        [0:6]   joint_pos     arm joints normalised to [-1,1]
        [6:12]  joint_vel     arm joints scaled to [-1,1]
        [12:15] ee_pos        EE relative to robot base
        [15:16] gripper       openness [0=open, 1=closed]
        [16:19] fish1_pos     fish1 relative to robot base
        [19:20] fish1_yaw     fish1 yaw angle
        [20:23] fish2_pos     fish2 relative to robot base
        [23:24] fish2_yaw
        [24:27] fish3_pos     fish3 relative to robot base
        [27:28] fish3_yaw
        [28:32] prev_action   last action
        """
        joint_pos   = ObservationTermCfg(func=_o.joint_pos_obs)
        joint_vel   = ObservationTermCfg(func=_o.joint_vel_obs)
        ee_pos      = ObservationTermCfg(func=_o.ee_pos_rel,   noise={"std": 0.005})
        gripper     = ObservationTermCfg(func=_o.gripper_obs)
        fish1_pos   = ObservationTermCfg(func=_o.fish_pos_rel, params={"fish_key": "fish"},  noise={"std": 0.005})
        fish1_yaw   = ObservationTermCfg(func=_o.fish_yaw_obs, params={"fish_key": "fish"})
        fish2_pos   = ObservationTermCfg(func=_o.fish_pos_rel, params={"fish_key": "fish2"}, noise={"std": 0.005})
        fish2_yaw   = ObservationTermCfg(func=_o.fish_yaw_obs, params={"fish_key": "fish2"})
        fish3_pos   = ObservationTermCfg(func=_o.fish_pos_rel, params={"fish_key": "fish3"}, noise={"std": 0.005})
        fish3_yaw   = ObservationTermCfg(func=_o.fish_yaw_obs, params={"fish_key": "fish3"})
        prev_action = ObservationTermCfg(func=_o.prev_action_obs)

        enable_corruption: bool = True
        concatenate_terms: bool = True

    policy: PolicyCfg = PolicyCfg()


# =============================================================================
# Rewards: S3 defines all terms; S2/S1 disable subsets via None
# =============================================================================

@configclass
class S3RewardsCfg:
    # Stage 1: approach
    approach_coarse = RewardTermCfg(func=_r.approach_reward,       weight=2.0, params={"std": 0.40})
    approach_fine   = RewardTermCfg(func=_r.approach_reward,       weight=2.0, params={"std": 0.07})
    reach_sparse    = RewardTermCfg(func=_r.reach_sparse,          weight=3.0, params={"threshold": 0.05})
    # Stage 2: grasp + lift + transport + delivery
    grasp_reward    = RewardTermCfg(func=_r.grasp_reward,          weight=2.0, params={"threshold": 0.08})
    lift_reward     = RewardTermCfg(func=_r.lift_reward,           weight=4.0, params={"lift_height": 0.10})
    fish_velocity   = RewardTermCfg(func=_r.fish_velocity_reward,  weight=6.0)
    transport_coarse= RewardTermCfg(func=_r.transport_reward,      weight=6.0, params={"std": 0.30})
    transport_fine  = RewardTermCfg(func=_r.transport_reward,      weight=3.0, params={"std": 0.08})
    vel_axis_align  = RewardTermCfg(func=_r.velocity_axis_alignment_reward, weight=3.0)
    yaw_alignment   = RewardTermCfg(func=_r.yaw_alignment_reward,  weight=3.0, params={"pos_threshold": 0.25})
    delivery        = RewardTermCfg(func=_r.delivery_reward,       weight=5.0, params={"pos_threshold": 0.12, "align_threshold": 0.75})
    ee_sep_p        = RewardTermCfg(func=_r.ee_separation_penalty, weight=1.0, params={"threshold": 0.15})
    # Stage 3: fall penalty
    fish_fall_p     = RewardTermCfg(func=_r.fish_fall_penalty,     weight=0.5)
    # Always: regularisation
    action_rate_p   = RewardTermCfg(func=_r.action_rate_penalty,   weight=0.05)
    joint_lim_p     = RewardTermCfg(func=_r.joint_limit_penalty,   weight=0.5, params={"margin": 0.05})


# =============================================================================
# Terminations
# =============================================================================

@configclass
class S3TerminationsCfg:
    time_out      = TerminationTermCfg(func=_t.max_episode_length, time_out=True)
    all_delivered = TerminationTermCfg(func=_t.all_fish_delivered)
    all_lost      = TerminationTermCfg(func=_t.all_fish_lost)


# =============================================================================
# Events
# =============================================================================

@configclass
class EventsCfg:

    reset_scene = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
            ]),
            "position_range": (-0.03, 0.03),
            "velocity_range": (-0.01, 0.01),
        },
    )

    reset_fish1 = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("fish"),
            "pose_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "yaw": (-0.52, 0.52)},
            "velocity_range": {},
        },
    )

    reset_fish2 = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("fish2"),
            "pose_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "yaw": (-0.52, 0.52)},
            "velocity_range": {},
        },
    )

    reset_fish3 = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("fish3"),
            "pose_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "yaw": (-0.52, 0.52)},
            "velocity_range": {},
        },
    )

    randomise_fish1_physics = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("fish"),
            "static_friction_range":  (0.15, 0.25),
            "dynamic_friction_range": (0.10, 0.20),
            "restitution_range":      (0.02, 0.05),
            "num_buckets": 1,
        },
    )

    randomise_fish2_physics = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("fish2"),
            "static_friction_range":  (0.15, 0.25),
            "dynamic_friction_range": (0.10, 0.20),
            "restitution_range":      (0.02, 0.05),
            "num_buckets": 1,
        },
    )

    randomise_fish3_physics = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("fish3"),
            "static_friction_range":  (0.15, 0.25),
            "dynamic_friction_range": (0.10, 0.20),
            "restitution_range":      (0.02, 0.05),
            "num_buckets": 1,
        },
    )

    remove_delivered = EventTermCfg(
        func=_ev.remove_delivered_fish,
        mode="interval",
        interval_range_s=(_POLICY_DT, _POLICY_DT),
        params={"pos_threshold": 0.12, "align_threshold": 0.75},
    )


# =============================================================================
# Stage 3: full task
# =============================================================================

@configclass
class Fish3S3EnvCfg(ManagerBasedRLEnvCfg):
    """Full 3-fish task with fall penalty. Train last."""
    scene:        Fish3SceneCfg    = Fish3SceneCfg(num_envs=4096, env_spacing=2.5)
    actions:      ActionsCfg       = ActionsCfg()
    observations: ObservationsCfg  = ObservationsCfg()
    rewards:      S3RewardsCfg     = S3RewardsCfg()
    terminations: S3TerminationsCfg= S3TerminationsCfg()
    events:       EventsCfg        = EventsCfg()
    sim: SimulationCfg = SimulationCfg(dt=0.01, render_interval=2, gravity=(0.0, 0.0, -9.81))
    episode_length_s: float = 25.0

    def __post_init__(self):
        super().__post_init__()
        self.max_episode_length = math.ceil(self.episode_length_s / self.sim.dt)
        self.decimation = 2
        self.viewer.eye    = (1.8, 0.8, 1.5)
        self.viewer.lookat = (0.5, 0.0, 0.8)
        # Re-enable transport shaping for S3: penalises fish overshooting outfeed centre.
        # Spawn baseline (~0.32 constant) is acceptable here — robot already knows infeed.
        self.rewards.transport_coarse.weight = 2.0
        self.rewards.transport_fine.weight   = 1.0
        # Amplify fall penalty — overshooting off the downstream edge is the primary failure
        self.rewards.fish_fall_p.weight = 2.0


@configclass
class Fish3S3EnvCfg_PLAY(Fish3S3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs    = 4
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.events.randomise_fish1_physics = None
        self.events.randomise_fish2_physics = None
        self.events.randomise_fish3_physics = None
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


# =============================================================================
# Stage 3a: position-only delivery (no orientation requirement)
# =============================================================================

@configclass
class Fish3S3aEnvCfg(Fish3S3EnvCfg):
    """Stage 3a: single-fish push-to-outfeed, position-only delivery.

    Structural changes vs S3:
      1. Single fish: fish2/fish3 permanently parked (disable_gravity=True, z=5.0).
         Policy masters 1-fish delivery before tackling 3.
      2. Gripper disabled: gripper_action=None — pushing only, arm is 3-DOF action space.
         Gripper stays open (initial joint_pos=0). Simpler obs/action space.
      3. 10s episodes (500 policy steps): delivery at step 250 worth 200×0.99^250≈16 —
         credit assignment is now feasible vs ≈0.9 at 550 steps.
      4. Hard joint clamping: JointLimitSafeIKAction (in ActionsCfg) prevents runaway.
         joint_lim_p is now a soft "stay away" signal at weight=2.0.
      5. Bidirectional fish_velocity: reward for progress, penalty for backward motion.
         hover=0.2/step; push-toward=0.4/step; push-backward=0.0/step.
    """
    def __post_init__(self):
        super().__post_init__()

        # ── Single-fish training ─────────────────────────────────────────────
        for _key in ("fish2", "fish3"):
            _fcfg = getattr(self.scene, _key)
            _fcfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=0.5,
            )
            _fcfg.init_state = RigidObjectCfg.InitialStateCfg(
                pos=(_fcfg.init_state.pos[0], _fcfg.init_state.pos[1], _PARKED_Z),
                rot=(1.0, 0.0, 0.0, 0.0),
            )
        self.events.reset_fish2 = None
        self.events.reset_fish3 = None
        self.events.randomise_fish2_physics = None
        self.events.randomise_fish3_physics = None

        # ── Episode length ───────────────────────────────────────────────────
        self.episode_length_s = 10.0
        self.max_episode_length = math.ceil(self.episode_length_s / self.sim.dt)

        # ── Gripper removed ──────────────────────────────────────────────────
        self.actions.gripper_action = None

        # ── Delivery: position-only, generous threshold ──────────────────────
        self.rewards.delivery.params["align_threshold"] = 0.0
        self.rewards.delivery.params["pos_threshold"]   = 0.15
        self.events.remove_delivered.params["align_threshold"] = 0.0
        self.events.remove_delivered.params["pos_threshold"]   = 0.15

        # ── Reward structure ─────────────────────────────────────────────────
        self.rewards.transport_coarse = None
        self.rewards.transport_fine   = None
        self.rewards.approach_fine    = None
        self.rewards.reach_sparse     = None
        self.rewards.vel_axis_align   = None
        self.rewards.yaw_alignment    = None
        self.rewards.ee_sep_p         = None

        self.rewards.approach_coarse.weight = 0.2   # navigation gradient
        self.rewards.fish_velocity.weight   = 8.0   # increased: approach_coarse saturates, push signal must dominate
        self.rewards.delivery.weight        = 200.0
        self.rewards.fish_fall_p.weight     = 5.0   # raised from 0.5: penalise launch-off-table failures
        self.rewards.action_rate_p.weight   = 0.3
        self.rewards.joint_lim_p.weight     = 2.0   # soft signal; hard enforcement via JointLimitSafeIKAction


@configclass
class Fish3S3aEnvCfg_PLAY(Fish3S3aEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs    = 4
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False

        # Fix fish1 friction to training mid-range
        self.events.randomise_fish1_physics.params["static_friction_range"]  = (0.20, 0.20)
        self.events.randomise_fish1_physics.params["dynamic_friction_range"] = (0.15, 0.15)
        self.events.randomise_fish1_physics.params["restitution_range"]      = (0.02, 0.02)
        self.events.randomise_fish2_physics = None
        self.events.randomise_fish3_physics = None

        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_fish1.params["pose_range"] = {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "yaw": (-0.52, 0.52)}
        # Despawn fish near the far end of the extended outfeed table (y∈[-0.90, 0.00])
        self.events.remove_delivered.params = {"y_delivered": -0.80}
        # Conveyor belt: set constant -Y velocity on fish that have crossed onto the outfeed
        self.events.outfeed_conveyor = EventTermCfg(
            func=_ev.apply_outfeed_conveyor,
            mode="interval",
            interval_range_s=(_POLICY_DT, _POLICY_DT),
            params={"conveyor_speed": 1.0},
        )


# =============================================================================
# Stage 2: pick and place (no fall penalty)
# =============================================================================

@configclass
class Fish3S2EnvCfg(Fish3S3EnvCfg):
    """Stage 2: grasp + transport + delivery.

    vs S3: transport_coarse/fine removed (free spawn-based baseline, no robot gradient),
    ee_sep_p removed (too punishing during grasp exploration), fish_fall_p amplified
    (fish falling kills all approach rewards), delivery orientation relaxed to 0.5.
    """
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s   = 20.0
        self.max_episode_length = math.ceil(self.episode_length_s / self.sim.dt)
        # transport and orientation terms removed for S2
        self.rewards.transport_coarse = None
        self.rewards.transport_fine = None
        self.rewards.vel_axis_align = None
        self.rewards.yaw_alignment = None
        # ee_sep_p removed: too punishing during grasp exploration
        self.rewards.ee_sep_p = None
        # fish_fall_p kept and amplified: S2 is when robot first touches fish;
        # without this, fish get knocked off tables and all approach rewards collapse to 0
        self.rewards.fish_fall_p.weight = 2.0
        self.rewards.delivery.params["align_threshold"] = 0.5


@configclass
class Fish3S2EnvCfg_PLAY(Fish3S2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs    = 4
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.events.randomise_fish1_physics = None
        self.events.randomise_fish2_physics = None
        self.events.randomise_fish3_physics = None
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


# =============================================================================
# Stage 1: approach only
# =============================================================================

@configclass
class Fish3S1EnvCfg(Fish3S2EnvCfg):
    """Stage 1: approach nearest fish. No grasping, no delivery."""
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s   = 8.0
        self.max_episode_length = math.ceil(self.episode_length_s / self.sim.dt)
        # Disable stage 2+ rewards
        self.rewards.fish_velocity     = None
        self.rewards.vel_axis_align    = None
        self.rewards.yaw_alignment     = None
        self.rewards.transport_coarse  = None
        self.rewards.transport_fine    = None
        self.rewards.delivery          = None
        self.rewards.ee_sep_p          = None
        # Disable delivery event and terminations (robot doesn't touch fish in S1)
        self.events.remove_delivered    = None
        self.terminations.all_delivered = None
        self.terminations.all_lost      = None


@configclass
class Fish3S1EnvCfg_PLAY(Fish3S1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs    = 4
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.events.randomise_fish1_physics = None
        self.events.randomise_fish2_physics = None
        self.events.randomise_fish3_physics = None
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


# =============================================================================
# Grasp-and-Place: single fish, full gripper, pick-and-place redesign
# =============================================================================

@configclass
class Fish3GraspEnvCfg(Fish3S3EnvCfg):
    """Single-fish grasp-and-place applying all lessons from S3a push experiments.

    Key design decisions:
      1. Gripper re-enabled: 4-DOF action space (3D position + 1D closure).
      2. lift_reward: rewards picking fish off the table — prevents drag-along-surface
         behaviour that satisfied S3a's fish_velocity signal without genuine pick-and-place.
      3. No Gaussian hover attractors: approach_fine, transport_coarse/fine, reach_sparse,
         vel_axis_align, yaw_alignment all removed.
      4. 15s episodes (750 policy steps): V_delivery ≈ 200 × 0.99^375 ≈ 4.8 — workable
         vs ≈ 0.9 at the original 25s S3 episodes.
      5. JointLimitSafeIKAction already in ActionsCfg — hard joint clamping retained.
      6. Transfer from S1 checkpoint: approach prior carried over, grasp/lift learned fresh.
    """

    def __post_init__(self):
        super().__post_init__()

        # ── Single-fish training ─────────────────────────────────────────────
        for _key in ("fish2", "fish3"):
            _fcfg = getattr(self.scene, _key)
            _fcfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, max_depenetration_velocity=0.5,
            )
            _fcfg.init_state = RigidObjectCfg.InitialStateCfg(
                pos=(_fcfg.init_state.pos[0], _fcfg.init_state.pos[1], _PARKED_Z),
                rot=(1.0, 0.0, 0.0, 0.0),
            )
        self.events.reset_fish2 = None
        self.events.reset_fish3 = None
        self.events.randomise_fish2_physics = None
        self.events.randomise_fish3_physics = None

        # ── Episode length: 15 s ─────────────────────────────────────────────
        self.episode_length_s = 15.0
        self.max_episode_length = math.ceil(self.episode_length_s / self.sim.dt)

        # ── Delivery: position-only, 15 cm threshold ─────────────────────────
        self.rewards.delivery.params["align_threshold"] = 0.0
        self.rewards.delivery.params["pos_threshold"]   = 0.15
        self.events.remove_delivered.params["align_threshold"] = 0.0
        self.events.remove_delivered.params["pos_threshold"]   = 0.15

        # ── Remove hover attractors ──────────────────────────────────────────
        self.rewards.transport_coarse = None
        self.rewards.transport_fine   = None
        self.rewards.approach_fine    = None
        self.rewards.reach_sparse     = None
        self.rewards.vel_axis_align   = None
        self.rewards.yaw_alignment    = None

        # ── Reward weights ───────────────────────────────────────────────────
        self.rewards.approach_coarse.weight = 0.5   # navigation only; low to avoid hover
        self.rewards.grasp_reward.weight    = 3.0   # closure gradient when near fish
        self.rewards.lift_reward.weight     = 4.0   # pick up, don't drag
        self.rewards.fish_velocity.weight   = 3.0   # potential-based transport while holding
        self.rewards.ee_sep_p.weight        = 1.0   # closed gripper must stay near fish
        self.rewards.delivery.weight        = 200.0
        self.rewards.fish_fall_p.weight     = 1.0
        self.rewards.action_rate_p.weight   = 0.3
        self.rewards.joint_lim_p.weight     = 2.0


@configclass
class Fish3GraspEnvCfg_PLAY(Fish3GraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs    = 4
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.events.randomise_fish1_physics = None
        self.events.randomise_fish2_physics = None
        self.events.randomise_fish3_physics = None
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


# =============================================================================
# Stage 3b: three-fish push (extends S3a to all fish)
# =============================================================================

_FULL_YAW = (-3.14159, 3.14159)
_SPAWN_XY  = (-0.10, 0.10)

@configclass
class Fish3S3bEnvCfg(Fish3S3aEnvCfg):
    """Stage 3b: three-fish push-to-outfeed. Extends S3a with all fish active.

    Transfer from S3a checkpoint (same 31-dim obs / 3-DOF action space).
    Longer episodes (20 s) to allow delivering all three fish.
    Full yaw randomisation (±180°) and wider spawn area (±10 cm).
    """
    def __post_init__(self):
        super().__post_init__()

        # ── Re-enable fish2 and fish3 on the table ───────────────────────────
        for _key in ("fish2", "fish3"):
            _fcfg = getattr(self.scene, _key)
            _fcfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=0.5,
                linear_damping=0.5,
                angular_damping=1.0,
            )
            _fcfg.init_state = RigidObjectCfg.InitialStateCfg(
                pos=(_fcfg.init_state.pos[0], _fcfg.init_state.pos[1], _FISH_Z),
                rot=(1.0, 0.0, 0.0, 0.0),
            )
        self.events.reset_fish2 = EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("fish2"),
                "pose_range": {"x": _SPAWN_XY, "y": _SPAWN_XY, "yaw": _FULL_YAW},
                "velocity_range": {},
            },
        )
        self.events.reset_fish3 = EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("fish3"),
                "pose_range": {"x": _SPAWN_XY, "y": _SPAWN_XY, "yaw": _FULL_YAW},
                "velocity_range": {},
            },
        )
        self.events.randomise_fish2_physics = EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("fish2"),
                "static_friction_range":  (0.15, 0.25),
                "dynamic_friction_range": (0.10, 0.20),
                "restitution_range":      (0.02, 0.05),
                "num_buckets": 1,
            },
        )
        self.events.randomise_fish3_physics = EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("fish3"),
                "static_friction_range":  (0.15, 0.25),
                "dynamic_friction_range": (0.10, 0.20),
                "restitution_range":      (0.02, 0.05),
                "num_buckets": 1,
            },
        )

        # ── Widen fish1 spawn range too ──────────────────────────────────────
        self.events.reset_fish1.params["pose_range"] = {
            "x": _SPAWN_XY, "y": _SPAWN_XY, "yaw": _FULL_YAW,
        }

        # ── lift_reward is meaningless with no gripper — remove it ──────────
        self.rewards.lift_reward = None

        # ── Longer episodes for 3-fish delivery ─────────────────────────────
        self.episode_length_s   = 20.0
        self.max_episode_length = math.ceil(self.episode_length_s / self.sim.dt)


@configclass
class Fish3S3bEnvCfg_PLAY(Fish3S3bEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs    = 4
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.events.randomise_fish1_physics.params["static_friction_range"]  = (0.20, 0.20)
        self.events.randomise_fish1_physics.params["dynamic_friction_range"] = (0.15, 0.15)
        self.events.randomise_fish1_physics.params["restitution_range"]      = (0.02, 0.02)
        self.events.randomise_fish2_physics.params["static_friction_range"]  = (0.20, 0.20)
        self.events.randomise_fish2_physics.params["dynamic_friction_range"] = (0.15, 0.15)
        self.events.randomise_fish2_physics.params["restitution_range"]      = (0.02, 0.02)
        self.events.randomise_fish3_physics.params["static_friction_range"]  = (0.20, 0.20)
        self.events.randomise_fish3_physics.params["dynamic_friction_range"] = (0.15, 0.15)
        self.events.randomise_fish3_physics.params["restitution_range"]      = (0.02, 0.02)
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.remove_delivered.params = {"y_delivered": -0.80}
        self.events.outfeed_conveyor = EventTermCfg(
            func=_ev.apply_outfeed_conveyor,
            mode="interval",
            interval_range_s=(_POLICY_DT, _POLICY_DT),
            params={"conveyor_speed": 1.0},
        )


# =============================================================================
# Stage 3ab: two-fish intermediate (S3a → S3ab → S3b curriculum step)
# =============================================================================

_S3AB_YAW  = (-0.524, 0.524)   # ±30°
# Table: X ∈ [0.20, 1.00] (80 cm), Y ∈ [0.00, 0.90] (90 cm).
# fish1 init (0.40, 0.15): X range keeps it ≥0.30 m from left edge (0.20 m).
#   absolute X ∈ [0.30, 0.55], absolute Y ∈ [0.08, 0.55]
# fish2 init (0.80, 0.15): right X bound capped at +0.05 to stay ≥0.15 m
#   from the right table edge (1.00 m), preventing spawn-overhang falls.
#   absolute X ∈ [0.65, 0.85], absolute Y ∈ [0.08, 0.55]
# X regions never overlap; Y is fully independent for natural depth variation.
_S3AB_X1  = (-0.10, 0.15)    # fish1: clear of left edge
_S3AB_X2  = (-0.15, 0.05)    # fish2: clear of right edge
_S3AB_Y   = (-0.07, 0.40)    # asymmetric Y: covers front-to-mid table depth

@configclass
class Fish3S3abEnvCfg(Fish3S3aEnvCfg):
    """Two-fish intermediate stage between S3a (1 fish) and S3b (3 fish).

    Adds fish2 as a second active fish while fish3 remains parked.
    Spawn ranges are widened vs S3a (±15 cm X, −7/+40 cm Y) so the policy
    encounters the full range from fish far apart in table depth to side-by-side.
    Episode extended to 15 s to accommodate two sequential deliveries.

    Transfer from S3a checkpoint (same 31-dim obs / 3-DOF action space).
    Delivery ceiling: (2 × 200) / 750 steps ≈ 0.533.
    """
    def __post_init__(self):
        super().__post_init__()

        # ── Widen fish1 spawn to cover more of the infeed table ─────────────
        self.events.reset_fish1.params["pose_range"] = {
            "x": _S3AB_X1, "y": _S3AB_Y, "yaw": _S3AB_YAW,
        }

        # ── Activate fish2; fish3 stays parked (inherited from S3a) ─────────
        _fcfg2 = self.scene.fish2
        _fcfg2.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.5,
            linear_damping=0.5,
            angular_damping=1.0,
        )
        _fcfg2.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(_fcfg2.init_state.pos[0], _fcfg2.init_state.pos[1], _FISH_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        )

        # ── Reset fish2 with edge-safe spawn range ───────────────────────────
        self.events.reset_fish2 = EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("fish2"),
                "pose_range": {"x": _S3AB_X2, "y": _S3AB_Y, "yaw": _S3AB_YAW},
                "velocity_range": {},
            },
        )

        # ── Physics randomisation for fish2 ─────────────────────────────────
        self.events.randomise_fish2_physics = EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("fish2"),
                "static_friction_range":  (0.15, 0.25),
                "dynamic_friction_range": (0.10, 0.20),
                "restitution_range":      (0.02, 0.05),
                "num_buckets": 1,
            },
        )

        # ── gripper rewards are meaningless with no gripper ─────────────────
        self.rewards.grasp_reward = None
        self.rewards.lift_reward  = None

        # ── Extend episode for two deliveries ───────────────────────────────
        self.episode_length_s   = 15.0
        self.max_episode_length = math.ceil(self.episode_length_s / self.sim.dt)


@configclass
class Fish3S3abEnvCfg_PLAY(Fish3S3abEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs    = 4
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.events.randomise_fish1_physics.params["static_friction_range"]  = (0.20, 0.20)
        self.events.randomise_fish1_physics.params["dynamic_friction_range"] = (0.15, 0.15)
        self.events.randomise_fish1_physics.params["restitution_range"]      = (0.02, 0.02)
        self.events.randomise_fish2_physics.params["static_friction_range"]  = (0.20, 0.20)
        self.events.randomise_fish2_physics.params["dynamic_friction_range"] = (0.15, 0.15)
        self.events.randomise_fish2_physics.params["restitution_range"]      = (0.02, 0.02)
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.remove_delivered.params = {"y_delivered": -0.80}
        self.events.outfeed_conveyor = EventTermCfg(
            func=_ev.apply_outfeed_conveyor,
            mode="interval",
            interval_range_s=(_POLICY_DT, _POLICY_DT),
            params={"conveyor_speed": 1.0},
        )
