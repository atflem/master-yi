# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
Articulation configuration for UR10e + Robotiq 2F-140 gripper.
Targets: Isaac Lab 2.x / Isaac Sim 5.0  |  Floor-mounted.

This config is derived directly from the working ur_gripper.py base file.
Key points:
  - Uses the plain UR10e USD from ISAAC_NUCLEUS_DIR with the Robotiq variant
    applied via spawn.variants = {"Gripper": "Robotiq_2f_140"}.
  - disable_gravity=True on the robot (standard for manipulation arms in IsaacLab).
  - Three actuator groups matching the working ur_gripper.py structure:
      "shoulder" / "elbow" / "wrist"  — arm joints, physics-tuned gains
      "gripper_drive"                  — finger_joint (primary driver)
      "gripper_finger"                 — inner finger joints (auxiliary)
      "gripper_passive"                — passive / pad / knuckle joints
  - effort_limit_sim / velocity_limit_sim set on gripper groups only
    (arm groups omit them to use the uncapped solver defaults, matching
    the working config pattern in ur_gripper.py).
"""

from __future__ import annotations
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


UR10e_ROBOTIQ_CFG = ArticulationCfg(
    # prim_path is injected by .replace() in env_cfg.py — do not set here.
    spawn=sim_utils.UsdFileCfg(
        # The Nucleus UR10e USD is used here — NOT the custom UR-with-SuccGripper.usd.
        #
        # Why: the custom USD has "World" as its defaultPrim. When Isaac Lab spawns
        # the robot at {ENV_REGEX_NS}/Robot, the USD's defaultPrim is placed at that
        # path, so arm links end up at Robot/Robot/wrist_3_link (one level too deep).
        # Sensor prim paths like {ENV_REGEX_NS}/Robot/wrist_3_link then find nothing.
        #
        # The Nucleus USD's defaultPrim IS the articulation root, so arm links land
        # directly at Robot/base_link, Robot/wrist_3_link, etc. — matching every
        # prim_path used in the scene config and sensors.
        #
        # The Robotiq 2F-140 gripper is added via the built-in USD variant:
        #   variants={"Gripper": "Robotiq_2f_140"}
        # This is identical to how ur_gripper.py builds UR10e_ROBOTIQ_GRIPPER_CFG.
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10e/ur10e.usd",
        variants={"Gripper": "Robotiq_2f_140"},
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,          # standard for fixed-base arms in IsaacLab
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # True: PhysX enforces that arm links cannot penetrate each other.
            # False (previous): the IK solver could fold the arm through itself
            # because there was no physical constraint preventing link overlap.
            enabled_self_collisions=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=True,     # needed for fingertip contact rewards
    ),

    # ---- Initial pose ----
    # "Ready" pose: arm pointing forward and slightly down so the TCP is
    # roughly 50 cm in front of the base at ~waist height.
    # shoulder_pan=π keeps the arm pointing away from the robot base (forward).
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),               # floor-mounted; no z offset
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # Standard UR10e "ready" pose — this is the only pose that has been
            # confirmed to produce a correct, non-floor-penetrating TCP position.
            # With the FrameTransformer offset (0, 0, 0.175) in wrist_3_link's
            # LOCAL frame, any change to elbow or wrist angles can rotate that
            # frame so the offset points into the floor at rest, causing the
            # ee_workspace termination to fire at step 0.
            # DO NOT change elbow or wrist_1 without verifying FK live.
            #
            # shoulder_pan=0 : arm faces the cube zone directly (+X).
            # π/2 (90°) was tried but caused elbow-flip self-collisions: the IK
            # had to sweep 90° of shoulder rotation AND reach down to the floor
            # simultaneously, passing through a singularity.  With
            # enabled_self_collisions=True the arm physically jammed against
            # itself within a few steps, terminating every episode before any
            # learning could occur.
            # At 0°, IK only needs to lower the arm; the cube y-randomisation
            # (±0.15 m) still forces lateral TCP correction without requiring
            # a singularity-prone pan sweep.
            "shoulder_pan_joint":   0.0,
            "shoulder_lift_joint":  -math.pi / 2,   # -90°
            "elbow_joint":           math.pi / 2,   # +90°
            "wrist_1_joint":        -math.pi / 2,   # -90°
            "wrist_2_joint":        -math.pi / 2,   # -90°
            "wrist_3_joint":         0.0,
            # Gripper — fully open
            "finger_joint":                  0.0,
            ".*_inner_finger_joint":         0.0,
            ".*_inner_finger_pad_joint":     0.0,
            ".*_outer_.*_joint":             0.0,
        },
        joint_vel={".*": 0.0},
    ),

    # ---- Actuators ----
    # Gains are taken directly from the working ur_gripper.py.
    # The arm is split into three groups (shoulder / elbow / wrist) because
    # each segment has different inertia and tuned PD values.
    actuators={
        # --- Arm ---
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*"],
            stiffness=1320.0,
            damping=72.6636085,
            friction=0.0,
            armature=0.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            stiffness=600.0,
            damping=34.64101615,
            friction=0.0,
            armature=0.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"],
            stiffness=216.0,
            damping=29.39387691,
            friction=0.0,
            armature=0.0,
        ),
        # --- Gripper ---
        # Primary driver joint — controls open/close
        "gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=1.0,
            stiffness=11.25,
            damping=0.1,
            friction=0.0,
            armature=0.0,
        ),
        # Auxiliary inner-finger joints — follow the driver via USD mimic, but
        # need an actuator entry so IsaacLab can set their initial state.
        "gripper_finger": ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_finger_joint"],
            effort_limit_sim=1.0,
            velocity_limit_sim=1.0,
            stiffness=0.2,
            damping=0.001,
            friction=0.0,
            armature=0.0,
        ),
        # Passive joints — zero stiffness/damping, follow kinematics only
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_inner_finger_pad_joint",
                ".*_outer_finger_joint",
                "right_outer_knuckle_joint",
            ],
            effort_limit_sim=1.0,
            velocity_limit_sim=1.0,
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
        ),
    },
)