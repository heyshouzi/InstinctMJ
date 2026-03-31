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
# Matching IsaacLab T1_CFG init_state
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
# T1 Actuator Configurations - Matching IsaacLab T1_CFG
# ============================================================================

# Arm: Encos4310 (Gear 36:1, Rated 10Nm, Peak 30Nm, Rated 147rpm)
T1_DELAYED_ARM = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=[
            ".*_Shoulder_Pitch",
            ".*_Shoulder_Roll",
            ".*_Elbow_Pitch",
            ".*_Elbow_Yaw",
        ],
        velocity_limit=17.59291886010284,
        stiffness=27.884395922309743,
        damping=2.6627636677002515,
        effort_limit=38.3,
        armature=0.0282528,
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

# Waist: Encos6408-40T (Gear 25:1, Rated 13Nm, Peak 40Nm)
T1_DELAYED_WAIST = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=["Waist"],
        velocity_limit=14.660765716752367,
        stiffness=47.1890460427085,
        damping=4.50622196249286,
        effort_limit=68.0,
        armature=0.0478125,
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

# Legs: Hip-Pitch, Hip-Roll, Hip-Yaw, Knee-Pitch
T1_DELAYED_LEGS = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=[
            ".*_Hip_Pitch",
            ".*_Hip_Roll",
            ".*_Hip_Yaw",
            ".*_Knee_Pitch",
        ],
        velocity_limit={
            ".*_Hip_Pitch": 16.755160819145562,
            ".*_Hip_Roll": 14.660765716752367,
            ".*_Hip_Yaw": 14.660765716752367,
            ".*_Knee_Pitch": 14.660765716752367,
        },
        stiffness={
            ".*_Hip_Pitch": 51.707647025659234,
            ".*_Hip_Roll": 47.1890460427085,
            ".*_Hip_Yaw": 47.1890460427085,
            ".*_Knee_Pitch": 40.17399573981213,
        },
        damping={
            ".*_Hip_Pitch": 4.937716571870764,
            ".*_Hip_Roll": 4.50622196249286,
            ".*_Hip_Yaw": 4.50622196249286,
            ".*_Knee_Pitch": 4.795417504307883,
        },
        effort_limit={
            ".*_Hip_Pitch": 96.0,
            ".*_Hip_Roll": 68.0,
            ".*_Hip_Yaw": 68.0,
            ".*_Knee_Pitch": 125.0,
        },
        armature={
            ".*_Hip_Pitch": 0.0523908,
            ".*_Hip_Roll": 0.0478125,
            ".*_Hip_Yaw": 0.0478125,
            ".*_Knee_Pitch": 0.0636012,
        },
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

# Feet: Ankle-Pitch, Ankle-Roll
T1_DELAYED_FEET = DelayedInstinctActuatorCfg(
    base_cfg=InstinctActuatorCfg(
        target_names_expr=[
            ".*_Ankle_Pitch",
            ".*_Ankle_Roll",
        ],
        velocity_limit={
            ".*_Ankle_Pitch": 12.56637061435917,
            ".*_Ankle_Roll": 12.56637061435917,
        },
        stiffness={
            ".*_Ankle_Pitch": 67.02487827197386,
            ".*_Ankle_Roll": 15.080597611194122,
        },
        damping={
            ".*_Ankle_Pitch": 8.53387254969377,
            ".*_Ankle_Roll": 1.4400909927608239,
        },
        effort_limit={
            ".*_Ankle_Pitch": 76.0,
            ".*_Ankle_Roll": 76.0,
        },
        armature={
            ".*_Ankle_Pitch": 0.0679104,
            ".*_Ankle_Roll": 0.01527984,
        },
    ),
    delay_target="position",
    delay_min_lag=1,
    delay_max_lag=3,
)

t1_delayed_actuator_cfgs: tuple[ActuatorCfg, ...] = (
    T1_DELAYED_ARM,
    T1_DELAYED_WAIST,
    T1_DELAYED_LEGS,
    T1_DELAYED_FEET,
)


# Action scale for T1
T1_ACTION_SCALE: dict[str, float] = {}
for actuator_cfg in t1_delayed_actuator_cfgs:
    effort = actuator_cfg.base_cfg.effort_limit
    stiffness = actuator_cfg.base_cfg.stiffness
    if effort is None or stiffness == 0.0:
        continue
    if isinstance(effort, dict):
        for joint_expr in actuator_cfg.base_cfg.target_names_expr:
            # Get the effort for this joint expression pattern
            for pattern, val in effort.items():
                if any(joint_expr in p or p in joint_expr for p in [pattern]):
                    T1_ACTION_SCALE[joint_expr] = 0.25 * val / (stiffness.get(pattern, stiffness) if isinstance(stiffness, dict) else stiffness)
                    break
    else:
        for joint_expr in actuator_cfg.base_cfg.target_names_expr:
            T1_ACTION_SCALE[joint_expr] = 0.25 * effort / stiffness


T1_23DOF_CFG = EntityCfg(
    init_state=copy.deepcopy(_T1_INIT_STATE),
    spec_fn=get_t1_spec,
    articulation=EntityArticulationInfoCfg(
        actuators=tuple(copy.deepcopy(act) for act in t1_delayed_actuator_cfgs),
        soft_joint_pos_limit_factor=0.9,
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
]
