"""VMAP (WMO/M2 collision) stack: BIH, WorldModel and StaticMapTree."""

from core.terrain.vmap.bih import BIH
from core.terrain.vmap.tree import (
    IGNORE_M2,
    IGNORE_NOTHING,
    INTERNAL_REP_MID,
    ModelInstance,
    ModelSpawn,
    StaticMapTree,
    to_internal,
    to_world,
)
from core.terrain.vmap.world_model import (
    GroupModel,
    WmoLiquid,
    WorldModel,
    WorldModelStore,
)

__all__ = [
    "BIH",
    "IGNORE_M2",
    "IGNORE_NOTHING",
    "INTERNAL_REP_MID",
    "GroupModel",
    "ModelInstance",
    "ModelSpawn",
    "StaticMapTree",
    "WmoLiquid",
    "WorldModel",
    "WorldModelStore",
    "to_internal",
    "to_world",
]
