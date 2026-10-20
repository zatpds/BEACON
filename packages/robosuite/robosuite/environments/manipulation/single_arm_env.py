"""
Compatibility shim for robosuite 1.5.2.

In robosuite <=1.4.x, SingleArmEnv was a subclass of ManipulationEnv that
handled single-arm-specific setup. In 1.5.x that logic was folded into
ManipulationEnv, so we provide a thin wrapper here for packages (e.g. mimicgen)
that still import the old path.
"""
from copy import deepcopy

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv


def _upgrade_controller_config(cfg):
    """Convert a robosuite <=1.4.x flat controller config to the 1.5.x composite format."""
    if cfg is None:
        return None
    cfg = deepcopy(cfg)
    if cfg.get("type") in ("BASIC", "HYBRID_MOBILE_BASE", "WHOLE_BODY_IK"):
        return cfg
    inner = cfg
    gripper_cfg = inner.pop("gripper", {"type": "GRIP"})
    inner["gripper"] = gripper_cfg
    return {
        "type": "BASIC",
        "body_parts": {
            "right": inner,
        },
    }


class SingleArmEnv(ManipulationEnv):
    """Thin compat wrapper: translates removed 1.4.x kwargs to their 1.5.x equivalents."""

    def __init__(self, mount_types="default", **kwargs):
        kwargs.setdefault("base_types", mount_types)
        if "controller_configs" in kwargs and kwargs["controller_configs"] is not None:
            kwargs["controller_configs"] = _upgrade_controller_config(kwargs["controller_configs"])
        super().__init__(**kwargs)
