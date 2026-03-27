from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from mjlab.actuator import BuiltinPositionActuatorCfg, DelayedActuatorCfg
from mjlab.actuator.xml_actuator import XmlPositionActuator

if TYPE_CHECKING:
    from mjlab.entity import Entity


@dataclass(kw_only=True)
class InstinctActuatorCfg(BuiltinPositionActuatorCfg):
    """Builtin position actuator config with joint velocity limit metadata."""

    velocity_limit: float


@dataclass(kw_only=True)
class DelayedInstinctActuatorCfg(DelayedActuatorCfg):
    """Delayed wrapper for position actuator cfg with velocity limit metadata."""

    base_cfg: InstinctActuatorCfg
    delay_target: Literal["position"] = "position"

    @property
    def velocity_limit(self) -> float:
        return self.base_cfg.velocity_limit


@dataclass(kw_only=True)
class XmlPositionActuatorCfgInstinct(BuiltinPositionActuatorCfg):
    """Builtin position actuator config for T1 that wraps XML <position> actuators.

    Inherits all fields from BuiltinPositionActuatorCfg (stiffness, damping,
    effort_limit, armature, etc.) but overrides build() to return XmlPositionActuator
    which wraps existing XML actuators rather than creating new ones.
    """

    velocity_limit: float = 0.0

    def build(
        self, entity: "Entity", target_ids: list[int], target_names: list[str]
    ):
        return XmlPositionActuator(self, entity, target_ids, target_names)


@dataclass(kw_only=True)
class DelayedXmlPositionActuatorCfg(DelayedActuatorCfg):
    """Delayed wrapper for XML position actuator cfg with velocity limit metadata."""

    base_cfg: XmlPositionActuatorCfgInstinct
    delay_target: Literal["position"] = "position"

    @property
    def velocity_limit(self) -> float:
        return self.base_cfg.velocity_limit

