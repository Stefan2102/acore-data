"""WorldModel (.vmo) parser + raycast (port of VMAP WorldModel/GroupModel).

File names are ``<spawn.name>.vmo`` (see WorldModelStore::AcquireModelInstance).
The model data is stored in the vmap "internal representation": the X and Y
axes are mirrored around the map centre (see VMapMgr2::convertPositionToInternalRep).
"""

from __future__ import annotations

import math
import os
import struct
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.terrain.vmap.bih import BIH
from core.terrain.vmap.math3d import (
    Vec3,
    aabb_contains,
    intersect_triangle,
    mat3_from_euler_zyx,
    mat3_inverse,
    mat3_transpose,
    mat3_vec,
)

VMAP_MAGIC = b"VMAP_4.8"
LIQUID_TILE_SIZE = 533.333 / 128.0

MOD_M2 = 1
MOD_WORLDSPAWN = 1 << 1
MOD_HAS_BOUND = 1 << 2


class WmoLiquid:
    __slots__ = ("tiles_x", "tiles_y", "corner", "liquid_type",
                 "height", "flags")

    def __init__(self) -> None:
        self.tiles_x = 0
        self.tiles_y = 0
        self.corner: Vec3 = (0.0, 0.0, 0.0)
        self.liquid_type = 0
        self.height: List[float] = []
        self.flags: List[int] = []

    @classmethod
    def read_from_file(cls, data: bytes, offset: int) -> Tuple["WmoLiquid", int]:
        liquid = cls()
        (tiles_x, tiles_y) = struct.unpack_from("<II", data, offset)
        corner = struct.unpack_from("<3f", data, offset + 8)
        (liquid_type,) = struct.unpack_from("<I", data, offset + 20)
        pos = offset + 24

        liquid.tiles_x = tiles_x
        liquid.tiles_y = tiles_y
        liquid.corner = corner
        liquid.liquid_type = liquid_type

        if tiles_x and tiles_y:
            size = (tiles_x + 1) * (tiles_y + 1)
            liquid.height = list(struct.unpack_from(f"<{size}f", data, pos))
            pos += size * 4
            flag_count = tiles_x * tiles_y
            liquid.flags = list(struct.unpack_from(f"<{flag_count}B", data, pos))
            pos += flag_count
        else:
            (h,) = struct.unpack_from("<f", data, pos)
            liquid.height = [h]
            pos += 4
        return liquid, pos

    def get_liquid_height(self, pos: Vec3) -> Optional[float]:
        """Port of WmoLiquid::GetLiquidHeight (returns None when outside)."""
        if not self.flags:
            return self.height[0]

        tx_f = (pos[0] - self.corner[0]) / LIQUID_TILE_SIZE
        tx = int(tx_f)
        if tx_f < 0.0 or tx >= self.tiles_x:
            return None
        ty_f = (pos[1] - self.corner[1]) / LIQUID_TILE_SIZE
        ty = int(ty_f)
        if ty_f < 0.0 or ty >= self.tiles_y:
            return None

        if self.flags and (self.flags[tx + ty * self.tiles_x] & 0x0F) == 0x0F:
            return None
        if not self.height:
            return None

        dx = tx_f - float(tx)
        dy = ty_f - float(ty)
        row_offset = self.tiles_x + 1
        if dx > dy:  # case (a)
            sx = self.height[tx + 1 + ty * row_offset] - self.height[tx + ty * row_offset]
            sy = self.height[tx + 1 + (ty + 1) * row_offset] - self.height[tx + 1 + ty * row_offset]
            return self.height[tx + ty * row_offset] + dx * sx + dy * sy
        # case (b)
        sx = self.height[tx + 1 + (ty + 1) * row_offset] - self.height[tx + (ty + 1) * row_offset]
        sy = self.height[tx + (ty + 1) * row_offset] - self.height[tx + ty * row_offset]
        return self.height[tx + ty * row_offset] + dx * sx + dy * sy


INSIDE = 0
MAYBE_INSIDE = 1
ABOVE = 2
OUT_OF_BOUNDS = -1


class GroupModel:
    __slots__ = ("bound_lo", "bound_hi", "mogp_flags", "group_wmo_id",
                 "vertices", "triangles", "mesh_tree", "liquid")

    def __init__(self) -> None:
        self.bound_lo: Vec3 = (0.0, 0.0, 0.0)
        self.bound_hi: Vec3 = (0.0, 0.0, 0.0)
        self.mogp_flags = 0
        self.group_wmo_id = 0
        self.vertices: List[Vec3] = []
        self.triangles: List[Tuple[int, int, int]] = []
        self.mesh_tree = BIH()
        self.liquid: Optional[WmoLiquid] = None

    def read_from_file(self, data: bytes, offset: int) -> int:
        lo = struct.unpack_from("<3f", data, offset)
        hi = struct.unpack_from("<3f", data, offset + 12)
        self.bound_lo = lo
        self.bound_hi = hi
        (self.mogp_flags,) = struct.unpack_from("<I", data, offset + 24)
        (self.group_wmo_id,) = struct.unpack_from("<I", data, offset + 28)
        pos = offset + 32

        if data[pos:pos + 4] != b"VERT":
            raise ValueError("GroupModel: expected VERT chunk")
        (chunk_size,) = struct.unpack_from("<I", data, pos + 4)
        (count,) = struct.unpack_from("<I", data, pos + 8)
        pos += 12
        if not count:
            # Models without (collision) geometry end here.
            return pos

        self.vertices = list(struct.unpack_from(f"<{count * 3}f", data, pos))
        self.vertices = [
            (self.vertices[i], self.vertices[i + 1], self.vertices[i + 2])
            for i in range(0, len(self.vertices), 3)
        ]
        pos += count * 12
        if chunk_size != 4 + count * 12:
            raise ValueError("GroupModel: bad VERT chunk size")

        if data[pos:pos + 4] != b"TRIM":
            raise ValueError("GroupModel: expected TRIM chunk")
        (chunk_size,) = struct.unpack_from("<I", data, pos + 4)
        (count,) = struct.unpack_from("<I", data, pos + 8)
        pos += 12
        tris = list(struct.unpack_from(f"<{count * 3}I", data, pos)) if count else []
        self.triangles = [(tris[i], tris[i + 1], tris[i + 2]) for i in range(0, len(tris), 3)]
        pos += count * 12
        if chunk_size != 4 + count * 12:
            raise ValueError("GroupModel: bad TRIM chunk size")

        if data[pos:pos + 4] != b"MBIH":
            raise ValueError("GroupModel: expected MBIH chunk")
        pos = self.mesh_tree.read_from_file(data, pos + 4)

        if data[pos:pos + 4] != b"LIQU":
            raise ValueError("GroupModel: expected LIQU chunk")
        (chunk_size,) = struct.unpack_from("<I", data, pos + 4)
        pos += 8
        if chunk_size > 0:
            self.liquid, pos = WmoLiquid.read_from_file(data, pos)
        return pos

    # ---- queries ----

    def intersect_ray(self, origin: Vec3, direction: Vec3,
                      distance: float, stop_at_first_hit: bool):
        """Port of GroupModel::IntersectRay. Returns (hit, distance)."""
        if not self.triangles:
            return False, distance

        state = [False]

        def callback(entry: int, dist: float):
            hit, dist = intersect_triangle(
                self.vertices, self.triangles[entry], origin, direction, dist
            )
            if hit:
                state[0] = True
            return hit, dist

        _, distance = self.mesh_tree.intersect_ray(
            origin, direction, distance, stop_at_first_hit, callback
        )
        return state[0], distance

    def is_inside_object(self, origin: Vec3):
        """Port of GroupModel::IsInsideObject. Returns (result, z_dist)."""
        if not self.triangles:
            return OUT_OF_BOUNDS, 0.0
        lo, hi = self.bound_lo, self.bound_hi
        if not (origin[0] >= lo[0] and origin[1] >= lo[1] and origin[2] >= lo[2]
                and origin[0] <= hi[0] and origin[1] <= hi[1]):
            return OUT_OF_BOUNDS, 0.0

        mlo, mhi = self.mesh_tree.bounds_lo, self.mesh_tree.bounds_hi
        if mhi[2] >= origin[2]:
            dist = math.inf
            hit, dist = self.intersect_ray(origin, (0.0, 0.0, -1.0), dist, False)
            if hit:
                return INSIDE, dist - 0.1
            if aabb_contains(mlo, mhi, origin):
                return MAYBE_INSIDE, 0.0
        else:
            dist = math.inf
            delta = origin[2] - mhi[2]
            bumped = (origin[0], origin[1], origin[2] - delta)
            hit, dist = self.intersect_ray(bumped, (0.0, 0.0, -1.0), dist, False)
            if hit:
                return ABOVE, dist - 0.1 + delta
        return OUT_OF_BOUNDS, 0.0

    def get_liquid_level(self, pos: Vec3) -> Optional[float]:
        if self.liquid:
            return self.liquid.get_liquid_height(pos)
        return None

    @property
    def liquid_type(self) -> int:
        return self.liquid.liquid_type if self.liquid else 0


class WorldModel:
    __slots__ = ("root_wmo_id", "groups", "group_tree", "flags")

    def __init__(self) -> None:
        self.root_wmo_id = 0
        self.groups: List[GroupModel] = []
        self.group_tree = BIH()
        self.flags = 0

    @classmethod
    def read_file(cls, path) -> Optional["WorldModel"]:
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            return None
        try:
            return cls._parse(data)
        except (ValueError, struct.error, IndexError):
            return None

    @classmethod
    def _parse(cls, data: bytes) -> "WorldModel":
        if data[:8] != VMAP_MAGIC:
            raise ValueError("WorldModel: bad magic")
        pos = 8
        if data[pos:pos + 4] != b"WMOD":
            raise ValueError("WorldModel: expected WMOD chunk")
        model = cls()
        (model.root_wmo_id,) = struct.unpack_from("<I", data, pos + 8)
        # "WMOD" + chunkSize + RootWMOID (the chunkSize is not used by the
        # reader; the payload is exactly the root id).
        pos += 12

        if data[pos:pos + 4] == b"GMOD":
            (count,) = struct.unpack_from("<I", data, pos + 4)
            pos += 8
            for _ in range(count):
                group = GroupModel()
                pos = group.read_from_file(data, pos)
                model.groups.append(group)
            if data[pos:pos + 4] != b"GBIH":
                raise ValueError("WorldModel: expected GBIH chunk")
            pos = model.group_tree.read_from_file(data, pos + 4)
        return model

    def intersect_ray(self, origin: Vec3, direction: Vec3,
                      distance: float, stop_at_first_hit: bool,
                      ignore_m2: bool):
        """Port of WorldModel::IntersectRay. Returns (hit, distance)."""
        if ignore_m2 and (self.flags & MOD_M2):
            return False, distance
        if len(self.groups) == 1:
            return self.groups[0].intersect_ray(origin, direction, distance,
                                                stop_at_first_hit)
        state = [False]

        def callback(entry: int, dist: float):
            hit, dist = self.groups[entry].intersect_ray(
                origin, direction, dist, stop_at_first_hit
            )
            if hit:
                state[0] = True
            return hit, dist

        _, distance = self.group_tree.intersect_ray(
            origin, direction, distance, stop_at_first_hit, callback
        )
        return state[0], distance

    def get_location_info(self, p: Vec3, down: Vec3):
        """Port of WorldModel::GetLocationInfo.

        Returns (dist, root_id, hit_group) or None.
        """
        if not self.groups:
            return None

        hits = [None, None, None]
        dist = [self.group_tree.bounds_hi[0] - self.group_tree.bounds_lo[0],
                self.group_tree.bounds_hi[1] - self.group_tree.bounds_lo[1],
                self.group_tree.bounds_hi[2] - self.group_tree.bounds_lo[2]]
        z_dist = math.sqrt(dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2])

        ray_origin = (
            p[0] - down[0] * 0.1,
            p[1] - down[1] * 0.1,
            p[2] - down[2] * 0.1,
        )

        def callback(entry: int, distance: float):
            group = self.groups[entry]
            result, group_z = group.is_inside_object(ray_origin)
            if result != OUT_OF_BOUNDS:
                if result != MAYBE_INSIDE:
                    if group_z < distance:
                        hits[result] = group
                        distance = group_z
                        return True, distance
                else:
                    hits[result] = group
            return False, distance

        _, z_dist = self.group_tree.intersect_ray(
            ray_origin, down, z_dist, False, callback
        )
        if hits[INSIDE] is not None:
            return z_dist, self.root_wmo_id, hits[INSIDE]
        if hits[MAYBE_INSIDE] is not None and hits[ABOVE] is not None:
            return z_dist, self.root_wmo_id, hits[MAYBE_INSIDE]
        return None


class WorldModelStore:
    """Caches parsed WorldModels by spawn name (lazy, LRU-capped)."""

    def __init__(self, base_path, max_models: int = 256):
        self._base = Path(base_path)
        self._max_models = max_models
        self._cache: "OrderedDict[str, Optional[WorldModel]]" = OrderedDict()

    def acquire(self, name: str, flags: int = 0) -> Optional[WorldModel]:
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        model = WorldModel.read_file(self._base / (name + ".vmo"))
        if model is not None:
            model.flags = flags
        self._cache[name] = model
        if len(self._cache) > self._max_models:
            self._cache.popitem(last=False)
        return model

    def clear(self) -> None:
        self._cache.clear()


def vec_mat(v: Vec3, m) -> Vec3:
    """G3D ``v * M`` == ``M.transpose() * v``."""
    return mat3_vec(mat3_transpose(m), v)


def make_inv_rot(rot: Vec3):
    """Port of ModelInstance's iInvRot computation."""
    m = mat3_from_euler_zyx(
        math.pi * rot[1] / 180.0,
        math.pi * rot[0] / 180.0,
        math.pi * rot[2] / 180.0,
    )
    return mat3_inverse(m)
