"""StaticMapTree / ModelInstance port: vmap spawns, tiles and raycasts."""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.terrain.vmap.bih import BIH
from core.terrain.vmap.math3d import (
    Vec3,
    aabb_contains,
    mat3_vec,
    ray_intersection_time_aabb,
    v_add,
    v_length,
    v_mad,
    v_mul,
    v_sub,
)
from core.terrain.vmap.world_model import (
    MOD_HAS_BOUND,
    MOD_M2,
    VMAP_MAGIC,
    WorldModel,
    WorldModelStore,
    make_inv_rot,
    vec_mat,
)

# 0.5f * 64 * 533.3333f, rounded to float32 exactly as the server computes it.
INTERNAL_REP_MID = 17066.666015625

# VMAP::ModelIgnoreFlags
IGNORE_NOTHING = 0
IGNORE_M2 = 1

FLT_MAX = 3.4028234663852886e38


def to_internal(x: float, y: float, z: float) -> Vec3:
    """VMapMgr2::convertPositionToInternalRep."""
    return (INTERNAL_REP_MID - x, INTERNAL_REP_MID - y, z)


def to_world(x: float, y: float, z: float) -> Vec3:
    return (INTERNAL_REP_MID - x, INTERNAL_REP_MID - y, z)


class ModelSpawn:
    __slots__ = ("flags", "adt_id", "id", "pos", "rot", "scale",
                 "bound_lo", "bound_hi", "name")

    def __init__(self) -> None:
        self.flags = 0
        self.adt_id = 0
        self.id = 0
        self.pos: Vec3 = (0.0, 0.0, 0.0)
        self.rot: Vec3 = (0.0, 0.0, 0.0)
        self.scale = 1.0
        self.bound_lo: Vec3 = (0.0, 0.0, 0.0)
        self.bound_hi: Vec3 = (0.0, 0.0, 0.0)
        self.name = ""


def read_model_spawn(data: bytes, offset: int) -> Tuple[ModelSpawn, int]:
    spawn = ModelSpawn()
    (spawn.flags,) = struct.unpack_from("<I", data, offset)
    (spawn.adt_id,) = struct.unpack_from("<H", data, offset + 4)
    (spawn.id,) = struct.unpack_from("<I", data, offset + 6)
    spawn.pos = struct.unpack_from("<3f", data, offset + 10)
    spawn.rot = struct.unpack_from("<3f", data, offset + 22)
    (spawn.scale,) = struct.unpack_from("<f", data, offset + 34)
    pos = offset + 38
    if spawn.flags & MOD_HAS_BOUND:
        spawn.bound_lo = struct.unpack_from("<3f", data, pos)
        spawn.bound_hi = struct.unpack_from("<3f", data, pos + 12)
        pos += 24
    (name_len,) = struct.unpack_from("<I", data, pos)
    pos += 4
    if name_len > 500:
        raise ValueError("ModelSpawn: file name too long")
    # Some extracted spawn names carry a trailing NUL; C fopen() stops at the
    # first NUL, so mirror that by truncating there.
    raw_name = data[pos:pos + name_len].split(b"\x00", 1)[0]
    spawn.name = raw_name.decode("utf-8", errors="replace")
    pos += name_len
    return spawn, pos


class ModelInstance:
    __slots__ = ("spawn", "store", "_model", "_tried", "inv_rot", "inv_scale")

    def __init__(self, spawn: ModelSpawn, store: WorldModelStore) -> None:
        self.spawn = spawn
        self.store = store
        self._model: Optional[WorldModel] = None
        self._tried = False
        self.inv_rot = make_inv_rot(spawn.rot)
        self.inv_scale = 1.0 / spawn.scale if spawn.scale else 0.0

    def get_model(self) -> Optional[WorldModel]:
        if self._model is None and not self._tried:
            self._tried = True
            self._model = self.store.acquire(self.spawn.name, self.spawn.flags)
        return self._model

    @property
    def flags(self) -> int:
        return self.spawn.flags

    @property
    def adt_id(self) -> int:
        return self.spawn.adt_id

    @property
    def id(self) -> int:
        return self.spawn.id

    def intersect_ray(self, origin: Vec3, direction: Vec3, max_dist: float,
                      stop_at_first_hit: bool, ignore_m2: bool = False):
        model = self.get_model()
        if model is None:
            return False, max_dist
        spawn = self.spawn
        time = ray_intersection_time_aabb(origin, direction,
                                          spawn.bound_lo, spawn.bound_hi)
        if math.isinf(time):
            return False, max_dist

        p = mat3_vec(self.inv_rot, v_sub(origin, spawn.pos))
        p = v_mul(p, self.inv_scale)
        mod_dir = mat3_vec(self.inv_rot, direction)
        distance = max_dist * self.inv_scale
        hit, distance = model.intersect_ray(p, mod_dir, distance,
                                            stop_at_first_hit, ignore_m2)
        if hit:
            distance *= spawn.scale
            max_dist = distance
        return hit, max_dist

    def get_location_info(self, p: Vec3, info: Dict) -> bool:
        model = self.get_model()
        if model is None:
            return False
        if self.spawn.flags & MOD_M2:
            return False
        if not aabb_contains(self.spawn.bound_lo, self.spawn.bound_hi, p):
            return False

        p_model = mat3_vec(self.inv_rot, v_sub(p, self.spawn.pos))
        p_model = v_mul(p_model, self.inv_scale)
        z_dir_model = mat3_vec(self.inv_rot, (0.0, 0.0, -1.0))

        res = model.get_location_info(p_model, z_dir_model)
        if not res:
            return False
        z_dist, root_id, hit_group = res
        model_ground = v_mad(p_model, z_dir_model, z_dist)
        world = v_add(v_mul(vec_mat(model_ground, self.inv_rot), self.spawn.scale),
                      self.spawn.pos)
        world_z = world[2]
        if info["ground_z"] < world_z:
            info["root_id"] = root_id
            info["hit_group"] = hit_group
            info["ground_z"] = world_z
            info["hit_instance"] = self
            return True
        return False

    def get_liquid_level(self, p: Vec3, info: Dict) -> Optional[float]:
        if info.get("hit_group") is None:
            return None
        p_model = mat3_vec(self.inv_rot, v_sub(p, self.spawn.pos))
        p_model = v_mul(p_model, self.inv_scale)
        liq = info["hit_group"].get_liquid_level(p_model)
        if liq is None:
            return None
        world = v_add(
            v_mul(vec_mat((p_model[0], p_model[1], liq), self.inv_rot), self.spawn.scale),
            self.spawn.pos,
        )
        return world[2]


class StaticMapTree:
    """Static vmap tree for one map (lazy tile + model loading)."""

    def __init__(self, map_id: int, vmaps_dir, store: Optional[WorldModelStore] = None):
        self.map_id = map_id
        self.dir = Path(vmaps_dir)
        self.store = store or WorldModelStore(self.dir)
        self.tree = BIH()
        self.tree_values: List[Optional[ModelInstance]] = []
        self.is_tiled = False
        self._loaded_tiles: set = set()
        self._assigned: set = set()
        self._valid = False
        self._load_tree()

    def _load_tree(self) -> None:
        path = self.dir / f"{self.map_id:03d}.vmtree"
        try:
            data = path.read_bytes()
        except OSError:
            return
        if data[:8] != VMAP_MAGIC:
            return
        self.is_tiled = data[8] != 0
        pos = 9
        if data[pos:pos + 4] != b"NODE":
            return
        pos += 4
        pos = self.tree.read_from_file(data, pos)
        n_values = self.tree.prim_count
        self.tree_values = [None] * n_values
        if data[pos:pos + 4] != b"GOBJ":
            return
        pos += 4
        if not self.is_tiled:
            spawn, pos = read_model_spawn(data, pos)
            if n_values > 0 and self._model_exists(spawn.name):
                self.tree_values[0] = ModelInstance(spawn, self.store)
        self._valid = True

    def _model_exists(self, name: str) -> bool:
        return (self.dir / (name + ".vmo")).exists()

    @property
    def valid(self) -> bool:
        return self._valid

    def tile_path(self, gx: int, gy: int) -> Path:
        # StaticMapTree::getTileFileName writes tileY first.
        return self.dir / f"{self.map_id:03d}_{gy:02d}_{gx:02d}.vmtile"

    def load_tile(self, gx: int, gy: int) -> bool:
        key = (gx, gy)
        if key in self._loaded_tiles:
            return True
        try:
            data = self.tile_path(gx, gy).read_bytes()
        except OSError:
            self._loaded_tiles.add(key)
            return False
        if data[:8] != VMAP_MAGIC:
            self._loaded_tiles.add(key)
            return False
        (num_spawns,) = struct.unpack_from("<I", data, 8)
        pos = 12
        n_values = len(self.tree_values)
        for _ in range(num_spawns):
            spawn, pos = read_model_spawn(data, pos)
            (ref,) = struct.unpack_from("<I", data, pos)
            pos += 4
            if ref >= n_values:
                continue
            if ref in self._assigned or self.tree_values[ref] is not None:
                continue
            if not self._model_exists(spawn.name):
                continue
            self.tree_values[ref] = ModelInstance(spawn, self.store)
            self._assigned.add(ref)
        self._loaded_tiles.add(key)
        return True

    def ensure_tiles_for_point(self, world_x: float, world_y: float,
                               radius: int = 1) -> None:
        """Load the containing vmap tile and (optionally) its neighbours."""
        from core.terrain.coords import world_to_gridcoord
        c = world_to_gridcoord(world_x, world_y)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                gx, gy = c.x + dx, c.y + dy
                if 0 <= gx < 64 and 0 <= gy < 64:
                    self.load_tile(gx, gy)

    # ---- queries (internal coordinate space) ----

    def _get_intersection_time(self, origin: Vec3, direction: Vec3,
                               max_dist: float, stop_at_first_hit: bool,
                               ignore_flags: int):
        state = [False]

        def callback(entry: int, dist: float):
            instance = self.tree_values[entry] if entry < len(self.tree_values) else None
            if instance is None:
                return False, dist
            hit, dist = instance.intersect_ray(
                origin, direction, dist, stop_at_first_hit,
                bool(ignore_flags & IGNORE_M2),
            )
            if hit:
                state[0] = True
            return hit, dist

        _, max_dist = self.tree.intersect_ray(
            origin, direction, max_dist, stop_at_first_hit, callback
        )
        return state[0], max_dist

    def get_height(self, x: float, y: float, z: float,
                   max_search_dist: float = 50.0) -> float:
        """Port of StaticMapTree::getHeight (world coords in, world Z out)."""
        self.ensure_tiles_for_point(x, y)
        origin = to_internal(x, y, z)
        hit, dist = self._get_intersection_time(
            origin, (0.0, 0.0, -1.0), max_search_dist, False, IGNORE_NOTHING
        )
        if not hit:
            return math.inf
        return origin[2] - dist

    def get_object_hit_pos(self, p1: Vec3, p2: Vec3, modify_dist: float):
        """Port of StaticMapTree::GetObjectHitPos (internal coords)."""
        max_dist = v_length(v_sub(p2, p1))
        if max_dist >= FLT_MAX or not math.isfinite(max_dist):
            return False, p2
        if max_dist < 1e-10:
            return False, p2
        direction = v_mul(v_sub(p2, p1), 1.0 / max_dist)
        hit, dist = self._get_intersection_time(
            p1, direction, max_dist, False, IGNORE_NOTHING
        )
        if not hit:
            return False, p2
        result = v_mad(p1, direction, dist)
        if modify_dist < 0:
            if v_length(v_sub(result, p1)) > -modify_dist:
                result = v_mad(result, direction, modify_dist)
            else:
                result = p1
        else:
            result = v_mad(result, direction, modify_dist)
        return True, result

    def is_in_line_of_sight(self, p1: Vec3, p2: Vec3,
                            ignore_flags: int = IGNORE_NOTHING) -> bool:
        """Port of StaticMapTree::isInLineOfSight (internal coords)."""
        max_dist = v_length(v_sub(p2, p1))
        if max_dist >= FLT_MAX or not math.isfinite(max_dist):
            return False
        if max_dist < 1e-10:
            return True
        direction = v_mul(v_sub(p2, p1), 1.0 / max_dist)
        hit, _ = self._get_intersection_time(
            p1, direction, max_dist, True, ignore_flags
        )
        return not hit

    def get_location_info(self, pos: Vec3) -> Optional[Dict]:
        """Port of StaticMapTree::GetLocationInfo (internal coords)."""
        info = {
            "ground_z": -math.inf,
            "root_id": -1,
            "hit_group": None,
            "hit_instance": None,
        }

        def callback(entry: int):
            instance = self.tree_values[entry] if entry < len(self.tree_values) else None
            if instance is not None:
                instance.get_location_info(pos, info)

        self.tree.intersect_point(pos, callback)
        if info["hit_instance"] is None:
            return None
        return info
