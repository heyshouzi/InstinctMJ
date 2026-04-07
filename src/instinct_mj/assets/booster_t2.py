"""Customized Booster T2 robot asset definitions."""

from __future__ import annotations

import copy
import os

import mujoco
from mjlab.actuator import ActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

from instinct_mj.actuators import DelayedInstinctActuatorCfg, InstinctActuatorCfg

__file_dir__ = os.path.dirname(os.path.realpath(__file__))

# T2 resources directory
T2_RESOURCES_DIR: str = os.path.join(__file_dir__, "resources", "booster_t2")

# MJCF (XML) and URDF path
T2_MJCF_PATH: str = os.path.join(T2_RESOURCES_DIR, "T2_V11_URDF.xml")
T2_URDF_PATH: str = os.path.join(T2_RESOURCES_DIR, "T2_V11_URDF.urdf")
T2_MESHES_DIR: str = os.path.join(T2_RESOURCES_DIR, "meshes")


def get_t2_assets(meshdir: str | None) -> dict[str, bytes]:
    """Load local T2 mesh assets keyed with MuJoCo meshdir prefix."""
    assets: dict[str, bytes] = {}
    normalized_meshdir = meshdir.rstrip("/") if meshdir else None
    update_assets(assets, T2_MESHES_DIR, normalized_meshdir)
    return assets


def get_t2_spec() -> mujoco.MjSpec:
    """Load the local T2_V11_URDF.xml as MjSpec."""
    spec = mujoco.MjSpec.from_file(T2_MJCF_PATH)
    spec.assets = get_t2_assets(spec.meshdir)
    return spec


# ============================================================================
# Initial state for T2 robot (root: Trunk)
# NOTE: root height and joint defaults will be tuned after initial testing.
# ============================================================================
_T2_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.72),
    joint_pos={
        # Upper-body default pose
        ".*AL1_Joint": 0.0,
        ".*AL2_Joint": 0.0,
        ".*AL3_Joint": 0.0,
        "AL4_Joint": 0.0,
        "AL5_Joint": 0.0,
        "AL6_Joint": 0.0,
        "AL7_Joint": 0.0,
        ".*AR2_Joint": 0.0,
        ".*AR3_Joint": 0.0,
        "AR4_Joint": 0.0,
        "AR5_Joint": 0.0,
        "AR6_Joint": 0.0,
        "AR7_Joint": 0.0,
        # Lower-body default pose (similar to T1)
        ".*HipPitch": -0.2,
        ".*ShankPitch": 0.4,
        ".*AnkleCross": -0.2,
    },
    joint_vel={".*": 0.0},
)


# ============================================================================
# Booster T2 Motor Specifications
# ============================================================================
# Motor assignments based on Booster T2 hardware table:
# - Head: 灵足14Nm (Gear 10:1, Rated 14Nm, Peak 14Nm, Rated 315rpm)
# - Arm Shoulder (AL1/AR1): 自研7025 (Gear 25:1, Rated 107Nm, Peak 107Nm, Rated 159rpm)
# - Arm Elbow (AL2-4/AR2-4): 自研5025 (Gear 25:1, Rated 35Nm, Peak 35Nm, Rated 135rpm)
# - Wrist (AL5-7/AR5-7): 高擎5036-02-JC交叉滚子 (Gear 36:1, Rated 20Nm, Peak 20Nm, Rated 75rpm)
# - Waist Yaw: 自研7025 (same as shoulder)
# - Waist Pitch/Roll: 因克斯4315 (Gear 36:1, Rated 75Nm, Peak 75Nm, Rated 120rpm)
# - Hip All (HipPitch/Roll/Yaw): 自研140NmB样 (Gear 22.5:1, Rated ~130Nm, Peak 130Nm, Rated 130rpm)
# - Knee (ShankPitch): 自研140NmB样 (same as hip)
# - Ankle (AnkleCross/FootRoll): 因克斯4315 (same as waist pitch/roll)
# ============================================================================

# ---- Motor output limits (peak torque, Nm) ----
ACTUATOR_LING14_EFFORT_LIMIT = 14.0  # 灵足14Nm
ACTUATOR_7025_EFFORT_LIMIT = 107.0  # 自研7025
ACTUATOR_5025_EFFORT_LIMIT = 35.0  # 自研5025
ACTUATOR_5036_EFFORT_LIMIT = 20.0  # 高擎5036-02
ACTUATOR_140NMB_EFFORT_LIMIT = 130.0  # 自研140NmB样
ACTUATOR_4315_EFFORT_LIMIT = 75.0  # 因克斯4315

# ---- Velocity limits (rad/s at motor shaft, before gear reduction) ----
VELOCITY_LIMIT_LING14 = 315 / 60 * 2 * 3.14159
VELOCITY_LIMIT_7025 = 159 / 60 * 2 * 3.14159
VELOCITY_LIMIT_5025 = 135 / 60 * 2 * 3.14159
VELOCITY_LIMIT_5036 = 75 / 60 * 2 * 3.14159
VELOCITY_LIMIT_140NMB = 130 / 60 * 2 * 3.14159
VELOCITY_LIMIT_4315 = 120 / 60 * 2 * 3.14159

# ---- Armature values (kg*m^2, motor-side, from motor constants table) ----
ARMATURE_LING14 = 0.001  # 灵足14Nm
ARMATURE_7025 = 0.02781875  # QDDH7025-25
ARMATURE_5025 = 0.0071  # QDDH5025-25
ARMATURE_5036 = 0.0132192  # 高擎5036-02
ARMATURE_140NMB = 0.030729032  # QDDH7629-22(140Nm)
ARMATURE_4315 = 0.0339552  # 因克斯4315

# ---- PD Control: G1 formula (BeyondMimic approach) ----
# NATURAL_FREQ = 10 * 2π  (10 Hz)
# DAMPING_RATIO = 2.0
# STIFFNESS = ARMATURE * NATURAL_FREQ^2
# DAMPING = 2 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

STIFFNESS_LING14 = ARMATURE_LING14 * NATURAL_FREQ**2
DAMPING_LING14 = 2.0 * DAMPING_RATIO * ARMATURE_LING14 * NATURAL_FREQ

STIFFNESS_7025 = ARMATURE_7025 * NATURAL_FREQ**2
DAMPING_7025 = 2.0 * DAMPING_RATIO * ARMATURE_7025 * NATURAL_FREQ

STIFFNESS_5025 = ARMATURE_5025 * NATURAL_FREQ**2
DAMPING_5025 = 2.0 * DAMPING_RATIO * ARMATURE_5025 * NATURAL_FREQ

STIFFNESS_5036 = ARMATURE_5036 * NATURAL_FREQ**2
DAMPING_5036 = 2.0 * DAMPING_RATIO * ARMATURE_5036 * NATURAL_FREQ

STIFFNESS_140NMB = ARMATURE_140NMB * NATURAL_FREQ**2
DAMPING_140NMB = 2.0 * DAMPING_RATIO * ARMATURE_140NMB * NATURAL_FREQ

STIFFNESS_4315 = ARMATURE_4315 * NATURAL_FREQ**2
DAMPING_4315 = 2.0 * DAMPING_RATIO * ARMATURE_4315 * NATURAL_FREQ


# ============================================================================
# T2 Actuator Configurations
# ============================================================================

# ---- Head: 灵足14Nm ----
T2_DELAYED_HEAD = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=("AAHeadYaw_Joint", "HeadPitch_Joint"),
        velocity_limit=VELOCITY_LIMIT_LING14,
        stiffness=STIFFNESS_LING14,
        damping=DAMPING_LING14,
        effort_limit=ACTUATOR_LING14_EFFORT_LIMIT,
        armature=ARMATURE_LING14,
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

# ---- Left Arm: Shoulder AL1 (7025) ----
T2_DELAYED_AL1 = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=("AL1_Joint",),
        velocity_limit=VELOCITY_LIMIT_7025,
        stiffness=STIFFNESS_7025,
        damping=DAMPING_7025,
        effort_limit=ACTUATOR_7025_EFFORT_LIMIT,
        armature=ARMATURE_7025,
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

# ---- Left Arm: AL2-4 (5025) ----
T2_DELAYED_AL234 = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=("AL2_Joint", "AL3_Joint", "AL4_Joint"),
        velocity_limit=VELOCITY_LIMIT_5025,
        stiffness=STIFFNESS_5025,
        damping=DAMPING_5025,
        effort_limit=ACTUATOR_5025_EFFORT_LIMIT,
        armature=ARMATURE_5025,
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

# ---- Left Arm: AL5-7 Wrist (5036) ----
T2_DELAYED_AL_WRIST = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=("AL5_Joint", "AL6_Joint", "AL7_Joint"),
        velocity_limit=VELOCITY_LIMIT_5036,
        stiffness=STIFFNESS_5036,
        damping=DAMPING_5036,
        effort_limit=ACTUATOR_5036_EFFORT_LIMIT,
        armature=ARMATURE_5036,
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

# ---- Right Arm: Shoulder AR1 (7025) ----
T2_DELAYED_AR1 = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=("AR1_Joint",),
        velocity_limit=VELOCITY_LIMIT_7025,
        stiffness=STIFFNESS_7025,
        damping=DAMPING_7025,
        effort_limit=ACTUATOR_7025_EFFORT_LIMIT,
        armature=ARMATURE_7025,
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

# ---- Right Arm: AR2-4 (5025) ----
T2_DELAYED_AR234 = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=("AR2_Joint", "AR3_Joint", "AR4_Joint"),
        velocity_limit=VELOCITY_LIMIT_5025,
        stiffness=STIFFNESS_5025,
        damping=DAMPING_5025,
        effort_limit=ACTUATOR_5025_EFFORT_LIMIT,
        armature=ARMATURE_5025,
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

# ---- Right Arm: AR5-7 Wrist (5036) ----
T2_DELAYED_AR_WRIST = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=("AR5_Joint", "AR6_Joint", "AR7_Joint"),
        velocity_limit=VELOCITY_LIMIT_5036,
        stiffness=STIFFNESS_5036,
        damping=DAMPING_5036,
        effort_limit=ACTUATOR_5036_EFFORT_LIMIT,
        armature=ARMATURE_5036,
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

# ---- Waist: Pitch/Roll (4315) ----
T2_DELAYED_WAIST_PITCH_ROLL = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=("WaistPitch_Joint", "WaistRoll_Joint"),
        velocity_limit=VELOCITY_LIMIT_4315,
        stiffness=STIFFNESS_4315,
        damping=DAMPING_4315,
        effort_limit=ACTUATOR_4315_EFFORT_LIMIT,
        armature=ARMATURE_4315,
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

# ---- Waist: Yaw (7025) ----
T2_DELAYED_WAIST_YAW = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=("WaistYaw_Joint",),
        velocity_limit=VELOCITY_LIMIT_7025,
        stiffness=STIFFNESS_7025,
        damping=DAMPING_7025,
        effort_limit=ACTUATOR_7025_EFFORT_LIMIT,
        armature=ARMATURE_7025,
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

# ---- Left Leg: Hip (140NmB) ----
T2_DELAYED_LEG_LEFT_HIP = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=("HipPitchLeft_Joint", "HipRollLeft_Joint", "HipYawLeft_Joint"),
        velocity_limit=VELOCITY_LIMIT_140NMB,
        stiffness=STIFFNESS_140NMB,
        damping=DAMPING_140NMB,
        effort_limit=ACTUATOR_140NMB_EFFORT_LIMIT,
        armature=ARMATURE_140NMB,
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

# ---- Left Leg: Knee (140NmB) ----
T2_DELAYED_LEG_LEFT_KNEE = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=("ShankPitchLeft_Joint",),
        velocity_limit=VELOCITY_LIMIT_140NMB,
        stiffness=STIFFNESS_140NMB,
        damping=DAMPING_140NMB,
        effort_limit=ACTUATOR_140NMB_EFFORT_LIMIT,
        armature=ARMATURE_140NMB,
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

# ---- Left Leg: Ankle (4315) ----
T2_DELAYED_LEG_LEFT_ANKLE = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=("AnkleCrossLeft_Joint", "FootLeft_Joint"),
        velocity_limit=VELOCITY_LIMIT_4315,
        stiffness=STIFFNESS_4315,
        damping=DAMPING_4315,
        effort_limit=ACTUATOR_4315_EFFORT_LIMIT,
        armature=ARMATURE_4315,
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

# ---- Right Leg: Hip (140NmB) ----
T2_DELAYED_LEG_RIGHT_HIP = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=("HipPitchRight_Joint", "HipRollRight_Joint", "HipYawRight_Joint"),
        velocity_limit=VELOCITY_LIMIT_140NMB,
        stiffness=STIFFNESS_140NMB,
        damping=DAMPING_140NMB,
        effort_limit=ACTUATOR_140NMB_EFFORT_LIMIT,
        armature=ARMATURE_140NMB,
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

# ---- Right Leg: Knee (140NmB) ----
T2_DELAYED_LEG_RIGHT_KNEE = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=("ShankPitchRight_Joint",),
        velocity_limit=VELOCITY_LIMIT_140NMB,
        stiffness=STIFFNESS_140NMB,
        damping=DAMPING_140NMB,
        effort_limit=ACTUATOR_140NMB_EFFORT_LIMIT,
        armature=ARMATURE_140NMB,
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

# ---- Right Leg: Ankle (4315) ----
T2_DELAYED_LEG_RIGHT_ANKLE = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=("AnkleCrossRight_Joint", "FootRight_Joint"),
        velocity_limit=VELOCITY_LIMIT_4315,
        stiffness=STIFFNESS_4315,
        damping=DAMPING_4315,
        effort_limit=ACTUATOR_4315_EFFORT_LIMIT,
        armature=ARMATURE_4315,
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

# ---- Full actuator list ----
t2_delayed_actuator_cfgs: tuple[ActuatorCfg, ...] = (
    T2_DELAYED_HEAD,
    T2_DELAYED_AL1,
    T2_DELAYED_AL234,
    T2_DELAYED_AL_WRIST,
    T2_DELAYED_AR1,
    T2_DELAYED_AR234,
    T2_DELAYED_AR_WRIST,
    T2_DELAYED_WAIST_PITCH_ROLL,
    T2_DELAYED_WAIST_YAW,
    T2_DELAYED_LEG_LEFT_HIP,
    T2_DELAYED_LEG_LEFT_KNEE,
    T2_DELAYED_LEG_LEFT_ANKLE,
    T2_DELAYED_LEG_RIGHT_HIP,
    T2_DELAYED_LEG_RIGHT_KNEE,
    T2_DELAYED_LEG_RIGHT_ANKLE,
)

# ---- Action scale: 0.25 * effort_limit / stiffness ----
T2_ACTION_SCALE: dict[str, float] = {}
for actuator_cfg in t2_delayed_actuator_cfgs:
    effort = actuator_cfg.base_cfg.effort_limit
    stiffness = actuator_cfg.base_cfg.stiffness
    if effort is None or stiffness == 0.0:
        continue
    for joint_expr in actuator_cfg.base_cfg.target_names_expr:
        T2_ACTION_SCALE[joint_expr] = 0.25 * effort / stiffness


T2_31DOF_CFG = EntityCfg(
    init_state=copy.deepcopy(_T2_INIT_STATE),
    spec_fn=get_t2_spec,
    articulation=EntityArticulationInfoCfg(
        actuators=tuple(copy.deepcopy(act) for act in t2_delayed_actuator_cfgs),
        soft_joint_pos_limit_factor=0.95,
    ),
)


# ============================================================================
# Symmetric augmentation for T2 (31 DOF)
# Joint order in URDF:
#  0: AAHeadYaw_Joint
#  1: HeadPitch_Joint
#  2: AL1_Joint         (left shoulder pitch)
#  3: AL2_Joint         (left shoulder roll)
#  4: AL3_Joint         (left shoulder yaw)
#  5: AL4_Joint         (left elbow)
#  6: AL5_Joint         (left wrist pitch)
#  7: AL6_Joint         (left wrist yaw)
#  8: AL7_Joint         (left wrist roll)
#  9: AR1_Joint         (right shoulder pitch)
# 10: AR2_Joint         (right shoulder roll)
# 11: AR3_Joint         (right shoulder yaw)
# 12: AR4_Joint         (right elbow)
# 13: AR5_Joint         (right wrist pitch)
# 14: AR6_Joint         (right wrist yaw)
# 15: AR7_Joint         (right wrist roll)
# 16: WaistPitch_Joint
# 17: WaistRoll_Joint
# 18: WaistYaw_Joint
# 19: HipPitchLeft_Joint
# 20: HipRollLeft_Joint
# 21: HipYawLeft_Joint
# 22: ShankPitchLeft_Joint
# 23: AnkleCrossLeft_Joint
# 24: FootLeft_Joint
# 25: HipPitchRight_Joint
# 26: HipRollRight_Joint
# 27: HipYawRight_Joint
# 28: ShankPitchRight_Joint
# 29: AnkleCrossRight_Joint
# 30: FootRight_Joint
# ============================================================================
T2_symmetric_augmentation_joint_mapping = [
    0,  # AAHeadYaw -> AAHeadYaw
    1,  # HeadPitch -> HeadPitch
    9,  # AL1 left -> AR1 right
    10,  # AL2 left -> AR2 right
    11,  # AL3 left -> AR3 right
    12,  # AL4 left -> AR4 right
    13,  # AL5 left -> AR5 right
    14,  # AL6 left -> AR6 right
    15,  # AL7 left -> AR7 right
    2,  # AR1 right -> AL1 left
    3,  # AR2 right -> AL2 left
    4,  # AR3 right -> AL3 left
    5,  # AR4 right -> AL4 left
    6,  # AR5 right -> AL5 left
    7,  # AR6 right -> AL6 left
    8,  # AR7 right -> AL7 left
    16,  # WaistPitch -> WaistPitch
    17,  # WaistRoll -> WaistRoll
    18,  # WaistYaw -> WaistYaw
    25,  # HipPitchLeft -> HipPitchRight
    26,  # HipRollLeft -> HipRollRight
    27,  # HipYawLeft -> HipYawRight
    28,  # ShankPitchLeft -> ShankPitchRight
    29,  # AnkleCrossLeft -> AnkleCrossRight
    30,  # FootLeft -> FootRight
    19,  # HipPitchRight -> HipPitchLeft
    20,  # HipRollRight -> HipRollLeft
    21,  # HipYawRight -> HipYawLeft
    22,  # ShankPitchRight -> ShankPitchLeft
    23,  # AnkleCrossRight -> AnkleCrossLeft
    24,  # FootRight -> FootLeft
]

T2_symmetric_augmentation_joint_reverse_buf = [
    1,  # AAHeadYaw
    1,  # HeadPitch
    1,  # AL1 (shoulder pitch)
    -1,  # AL2 (shoulder roll)
    1,  # AL3 (shoulder yaw)
    1,  # AL4 (elbow)
    1,  # AL5 (wrist pitch)
    -1,  # AL6 (wrist yaw)
    -1,  # AL7 (wrist roll)
    1,  # AR1
    -1,  # AR2
    1,  # AR3
    1,  # AR4
    1,  # AR5
    -1,  # AR6
    -1,  # AR7
    1,  # WaistPitch
    1,  # WaistRoll
    1,  # WaistYaw
    1,  # HipPitchLeft
    -1,  # HipRollLeft
    -1,  # HipYawLeft
    1,  # ShankPitchLeft (knee)
    1,  # AnkleCrossLeft
    -1,  # FootLeft (Roll - needs sign flip for left-right mirror)
    1,  # HipPitchRight
    -1,  # HipRollRight
    -1,  # HipYawRight
    1,  # ShankPitchRight (knee)
    1,  # AnkleCrossRight
    -1,  # FootRight (Roll - needs sign flip for left-right mirror)
]


__all__ = [
    "T2_MJCF_PATH",
    "T2_URDF_PATH",
    "T2_MESHES_DIR",
    "T2_31DOF_CFG",
    "T2_ACTION_SCALE",
    "T2_symmetric_augmentation_joint_mapping",
    "T2_symmetric_augmentation_joint_reverse_buf",
    "get_t2_assets",
    "get_t2_spec",
    "t2_delayed_actuator_cfgs",
    # Motor constants
    "ACTUATOR_LING14_EFFORT_LIMIT",
    "ACTUATOR_7025_EFFORT_LIMIT",
    "ACTUATOR_5025_EFFORT_LIMIT",
    "ACTUATOR_5036_EFFORT_LIMIT",
    "ACTUATOR_140NMB_EFFORT_LIMIT",
    "ACTUATOR_4315_EFFORT_LIMIT",
    "NATURAL_FREQ",
    "DAMPING_RATIO",
    "STIFFNESS_LING14",
    "STIFFNESS_7025",
    "STIFFNESS_5025",
    "STIFFNESS_5036",
    "STIFFNESS_140NMB",
    "STIFFNESS_4315",
    "DAMPING_LING14",
    "DAMPING_7025",
    "DAMPING_5025",
    "DAMPING_5036",
    "DAMPING_140NMB",
    "DAMPING_4315",
]
