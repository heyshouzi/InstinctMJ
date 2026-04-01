"""Customized Booster T1 robot asset definitions."""

from __future__ import annotations

import copy
import os

import mujoco
from mjlab.actuator import ActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

from instinct_mj.actuators import DelayedInstinctActuatorCfg, InstinctActuatorCfg

__file_dir__ = os.path.dirname(os.path.realpath(__file__))

# T1 resources directory
T1_RESOURCES_DIR: str = os.path.join(__file_dir__, "resources", "booster_t1")

# MJCF (XML) path – uses the Booster T1 23-dof model.
T1_MJCF_PATH: str = os.path.join(T1_RESOURCES_DIR, "T1_23dof.xml")
T1_MESHES_DIR: str = os.path.join(T1_RESOURCES_DIR, "meshes")


def get_t1_assets(meshdir: str | None) -> dict[str, bytes]:
    """Load local T1 mesh assets keyed with MuJoCo meshdir prefix."""
    assets: dict[str, bytes] = {}
    normalized_meshdir = meshdir.rstrip("/") if meshdir else None
    update_assets(assets, T1_MESHES_DIR, normalized_meshdir)
    return assets


def get_t1_spec() -> mujoco.MjSpec:
    """Load the local T1_23dof.xml as MjSpec."""
    spec = mujoco.MjSpec.from_file(T1_MJCF_PATH)
    spec.assets = get_t1_assets(spec.meshdir)
    return spec


# Initial state for T1 robot.
# NOTE: pos is the root (Trunk) world position.
_T1_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.70),
    joint_pos={
        # Upper-body default pose: arms hang down close to the torso
        ".*_Shoulder_Pitch": -0.1,
        "Left_Shoulder_Roll": -1.25,
        "Right_Shoulder_Roll": 1.25,
        ".*_Elbow_Pitch": 0.4,
        "Left_Elbow_Yaw": -0.6,
        "Right_Elbow_Yaw": 0.6,
        # Lower-body default pose
        ".*_Hip_Pitch": -0.2,
        ".*_Knee_Pitch": 0.4,
        ".*_Ankle_Pitch": -0.2,
    },
    joint_vel={".*": 0.0},
)


# ============================================================================
# Booster T1 Motor Specifications
# ============================================================================
# Motor assignments based on Booster T1 hardware:
# - Arm: Encos4310 (Gear 36:1, Rated 10Nm, Peak 30Nm, Rated 147rpm)
# - Ankle: Encos4315 (Gear 36:1, Rated 19Nm, Peak 57Nm, Rated 104rpm)
# - Waist, Hip-Roll & Hip-Yaw: Encos6408-40T non-standard (Gear 25:1, Rated 13Nm, Peak 40Nm)
# - Hip-Pitch: Encos8112 (Gear 18:1, Rated 30Nm, Peak 90Nm, Rated 140rpm)
# - Knee: Encos8116 (Gear 18:1, Rated 39Nm, Peak 118Nm, Rated 120rpm)
# - Neck: DMNA4310 (Gear 10:1, Rated 3Nm, Peak 7Nm, Rated 120rpm)
# ============================================================================

# Motor output limits based on IsaacLab T1_CFG
# Effort limits (matching IsaacLab T1_CFG values)
ACTUATOR_4310_EFFORT_LIMIT = 38.3  # Arm
ACTUATOR_4315_EFFORT_LIMIT = 76.0  # Ankle
ACTUATOR_6408_EFFORT_LIMIT = 68.0  # Waist, Hip_Roll, Hip_Yaw
ACTUATOR_8112_EFFORT_LIMIT = 96.0  # Hip_Pitch
ACTUATOR_8116_EFFORT_LIMIT = 125.0  # Knee_Pitch
ACTUATOR_DMNA4310_EFFORT_LIMIT = 38.3  # Head (uses arm actuator)

# Velocity limits (rad/s at MOTOR SHAFT - matching booster.py convention)
# These are motor-side velocities before gear reduction
# Calculated from rated rpm: rated_rpm / 60 * 2pi
VELOCITY_LIMIT_4310 = 147 / 60 * 2 * 3.14159  # ~15.4 rad/s
VELOCITY_LIMIT_4315 = 104 / 60 * 2 * 3.14159  # ~10.9 rad/s
VELOCITY_LIMIT_6408 = 57 / 60 * 2 * 3.14159  # ~6.0 rad/s
VELOCITY_LIMIT_8112 = 140 / 60 * 2 * 3.14159  # ~14.66 rad/s
VELOCITY_LIMIT_8116 = 120 / 60 * 2 * 3.14159  # ~12.57 rad/s
VELOCITY_LIMIT_DMNA4310 = 120 / 60 * 2 * 3.14159  # ~12.57 rad/s

# Armature values (from booster.py)
ARMATURE_4310 = 0.0282528
ARMATURE_4315 = 0.0339552
ARMATURE_6408 = 0.0478125
ARMATURE_8112 = 0.0523908
ARMATURE_8116 = 0.0636012
ARMATURE_DMNA4310 = 0.001

# Natural frequency and damping ratio for PD control
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

# Compute stiffness (k_p) and damping (k_d) from motor parameters
# k_p = k_t^2 / R_eff (motor-side), reflected through gear ratio
# Using rated torque and speed to estimate motor constants
# k_t = rated_torque / (rated_rpm / 60 * 2*pi)
# k_p_stiffness = k_t^2 / R * (1/gear_ratio)^2
# damping = 2 * damping_ratio * sqrt(k_p_stiffness * I_eff)

# Simplified approach: scale stiffness proportional to peak torque
# and normalize to a 90Nm reference (similar to G1 8112 motor)

# Stiffness values from IsaacLab T1_CFG
STIFFNESS_8116 = 40.17  # Knee
STIFFNESS_8112 = 51.71  # Hip-Pitch
STIFFNESS_4315 = 67.02  # Ankle_Pitch (pitch dominant)
STIFFNESS_6408 = 47.19  # Waist/Hip-Roll/Hip-Yaw
STIFFNESS_4310 = 27.88  # Arm
STIFFNESS_DMNA4310 = 27.88  # Head (uses arm actuator)

# Damping values from IsaacLab T1_CFG
DAMPING_8116 = 4.80
DAMPING_8112 = 4.94
DAMPING_4315 = 8.53
DAMPING_6408 = 4.51
DAMPING_4310 = 2.66
DAMPING_DMNA4310 = 2.66  # Head (uses arm actuator)


# ============================================================================
# T1 Actuator Configurations
# ============================================================================

# Hip-Pitch: Encos8112
T1_DELAYED_HIP_PITCH = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=(".*_Hip_Pitch",),
        velocity_limit=VELOCITY_LIMIT_8112,
        stiffness=STIFFNESS_8112,
        damping=DAMPING_8112,
        effort_limit=ACTUATOR_8112_EFFORT_LIMIT,
        armature=ARMATURE_8112,
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

# Knee: Encos8116
T1_DELAYED_KNEE = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=(".*_Knee_Pitch",),
        velocity_limit=VELOCITY_LIMIT_8116,
        stiffness=STIFFNESS_8116,
        damping=DAMPING_8116,
        effort_limit=ACTUATOR_8116_EFFORT_LIMIT,
        armature=ARMATURE_8116,
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

# Waist and Hip-Roll/Hip-Yaw: Encos6408-40T
T1_DELAYED_WAIST_HIP = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=("Waist", ".*_Hip_Roll", ".*_Hip_Yaw"),
        velocity_limit=VELOCITY_LIMIT_6408,
        stiffness=STIFFNESS_6408,
        damping=DAMPING_6408,
        effort_limit=ACTUATOR_6408_EFFORT_LIMIT,
        armature=ARMATURE_6408,
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

# Ankle: Encos4315
T1_DELAYED_ANKLE = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=(".*_Ankle_Pitch", ".*_Ankle_Roll"),
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

# Arm: Encos4310
T1_DELAYED_ARM = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=(
            ".*_Shoulder_Pitch",
            ".*_Shoulder_Roll",
            ".*_Elbow_Pitch",
            ".*_Elbow_Yaw",
        ),
        velocity_limit=VELOCITY_LIMIT_4310,
        stiffness=STIFFNESS_4310,
        damping=DAMPING_4310,
        effort_limit=ACTUATOR_4310_EFFORT_LIMIT,
        armature=ARMATURE_4310,
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

# Neck: DMNA4310 (Gear 10:1, Rated 3Nm, Peak 7Nm)
T1_DELAYED_NECK = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=("AAHead_yaw", "Head_pitch"),
        velocity_limit=VELOCITY_LIMIT_DMNA4310,
        stiffness=STIFFNESS_DMNA4310,
        damping=DAMPING_DMNA4310,
        effort_limit=ACTUATOR_DMNA4310_EFFORT_LIMIT,
        armature=ARMATURE_DMNA4310,
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

t1_delayed_actuator_cfgs: tuple[ActuatorCfg, ...] = (
    T1_DELAYED_HIP_PITCH,
    T1_DELAYED_KNEE,
    T1_DELAYED_WAIST_HIP,
    T1_DELAYED_ANKLE,
    T1_DELAYED_ARM,
    T1_DELAYED_NECK,
)


# Action scale for T1
T1_ACTION_SCALE: dict[str, float] = {}
for actuator_cfg in t1_delayed_actuator_cfgs:
    effort = actuator_cfg.base_cfg.effort_limit
    stiffness = actuator_cfg.base_cfg.stiffness
    if effort is None or stiffness == 0.0:
        continue
    for joint_expr in actuator_cfg.base_cfg.target_names_expr:
        T1_ACTION_SCALE[joint_expr] = 0.25 * effort / stiffness


T1_23DOF_CFG = EntityCfg(
    init_state=copy.deepcopy(_T1_INIT_STATE),
    spec_fn=get_t1_spec,
    articulation=EntityArticulationInfoCfg(
        actuators=tuple(copy.deepcopy(act) for act in t1_delayed_actuator_cfgs),
        soft_joint_pos_limit_factor=0.95,
    ),
)


# Symmetric augmentation for T1 (23 DOF)
# Joint order: Waist(0), Head_yaw(1), Head_pitch(2),
# Left arm: Shoulder_Pitch(3), Shoulder_Roll(4), Elbow_Pitch(5), Elbow_Yaw(6)
# Right arm: Shoulder_Pitch(7), Shoulder_Roll(8), Elbow_Pitch(9), Elbow_Yaw(10)
# Left leg: Hip_Pitch(11), Hip_Roll(12), Hip_Yaw(13), Knee_Pitch(14), Ankle_Pitch(15), Ankle_Roll(16)
# Right leg: Hip_Pitch(17), Hip_Roll(18), Hip_Yaw(19), Knee_Pitch(20), Ankle_Pitch(21), Ankle_Roll(22)
T1_symmetric_augmentation_joint_mapping = [
    0,  # Waist -> Waist (single joint)
    1,  # Head_yaw -> Head_yaw (single joint)
    2,  # Head_pitch -> Head_pitch (single joint)
    7,  # Left_Shoulder_Pitch -> Right_Shoulder_Pitch
    8,  # Left_Shoulder_Roll -> Right_Shoulder_Roll
    9,  # Left_Elbow_Pitch -> Right_Elbow_Pitch
    10,  # Left_Elbow_Yaw -> Right_Elbow_Yaw
    3,  # Right_Shoulder_Pitch -> Left_Shoulder_Pitch
    4,  # Right_Shoulder_Roll -> Left_Shoulder_Roll
    5,  # Right_Elbow_Pitch -> Left_Elbow_Pitch
    6,  # Right_Elbow_Yaw -> Left_Elbow_Yaw
    17,  # Left_Hip_Pitch -> Right_Hip_Pitch
    18,  # Left_Hip_Roll -> Right_Hip_Roll
    19,  # Left_Hip_Yaw -> Right_Hip_Yaw
    20,  # Left_Knee_Pitch -> Right_Knee_Pitch
    21,  # Left_Ankle_Pitch -> Right_Ankle_Pitch
    22,  # Left_Ankle_Roll -> Right_Ankle_Roll
    11,  # Right_Hip_Pitch -> Left_Hip_Pitch
    12,  # Right_Hip_Roll -> Left_Hip_Roll
    13,  # Right_Hip_Yaw -> Left_Hip_Yaw
    14,  # Right_Knee_Pitch -> Left_Knee_Pitch
    15,  # Right_Ankle_Pitch -> Left_Ankle_Pitch
    16,  # Right_Ankle_Roll -> Left_Ankle_Roll
]

T1_symmetric_augmentation_joint_reverse_buf = [
    1,  # Waist
    1,  # Head_yaw
    1,  # Head_pitch
    1,  # Left_Shoulder_Pitch
    -1,  # Left_Shoulder_Roll
    1,  # Left_Elbow_Pitch
    -1,  # Left_Elbow_Yaw
    1,  # Right_Shoulder_Pitch
    -1,  # Right_Shoulder_Roll
    1,  # Right_Elbow_Pitch
    -1,  # Right_Elbow_Yaw
    1,  # Left_Hip_Pitch
    -1,  # Left_Hip_Roll
    -1,  # Left_Hip_Yaw
    1,  # Left_Knee_Pitch
    1,  # Left_Ankle_Pitch
    -1,  # Left_Ankle_Roll
    1,  # Right_Hip_Pitch
    -1,  # Right_Hip_Roll
    -1,  # Right_Hip_Yaw
    1,  # Right_Knee_Pitch
    1,  # Right_Ankle_Pitch
    -1,  # Right_Ankle_Roll
]


__all__ = [
    "T1_MJCF_PATH",
    "T1_MESHES_DIR",
    "T1_23DOF_CFG",
    "T1_ACTION_SCALE",
    "T1_symmetric_augmentation_joint_mapping",
    "T1_symmetric_augmentation_joint_reverse_buf",
    "get_t1_assets",
    "get_t1_spec",
    "t1_delayed_actuator_cfgs",
    # Motor constants
    "ACTUATOR_4310_EFFORT_LIMIT",
    "ACTUATOR_4315_EFFORT_LIMIT",
    "ACTUATOR_6408_EFFORT_LIMIT",
    "ACTUATOR_8112_EFFORT_LIMIT",
    "ACTUATOR_8116_EFFORT_LIMIT",
    "ACTUATOR_DMNA4310_EFFORT_LIMIT",
    "STIFFNESS_8116",
    "STIFFNESS_8112",
    "STIFFNESS_4315",
    "STIFFNESS_6408",
    "STIFFNESS_4310",
    "STIFFNESS_DMNA4310",
]
