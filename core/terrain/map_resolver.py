"""Server-equivalent terrain queries: heights, liquids, areas and floors.

Ports the relevant AzerothCore Map/GridTerrainData logic (static data only;
dynamic game-object and transport collision is not reproducible offline):

  - Map::GetHeight           (ADT terrain + vmap raycast preference)
  - Map::GetFullTerrainStatusForPosition
  - Map::GetAreaId / GetZoneAndAreaId
  - Map::GetLiquidData       (WMO + grid liquids)
  - Map::GetWaterOrGroundLevel
  - WorldObject::UpdateAllowedPositionZ (unit profile)
  - Map2ZoneCoordinates
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from core.terrain.coords import get_data_paths, world_to_map_tile
from core.terrain.map_reader import (
    GRID_SIZE,
    GROUND_HEIGHT_TOLERANCE,
    INVALID_HEIGHT,
    LIQUID_TYPE_DARK_WATER,
    MAP_RESOLUTION,
    MapReader,
)
from core.terrain.vmap import (
    IGNORE_M2,
    IGNORE_NOTHING,
    StaticMapTree,
    to_internal,
    to_world,
)

VMAP_INVALID_HEIGHT = -100000.0
VMAP_INVALID_HEIGHT_VALUE = -200000.0
MAX_HEIGHT = 100000.0
Z_OFFSET_FIND_HEIGHT = 2.0
DEFAULT_HEIGHT_SEARCH = 50.0
DEFAULT_COLLISION_HEIGHT = 2.03128
DEFAULT_COLLISION_WIDTH = 2.03128

# LiquidData status (GridTerrainData.h)
LIQUID_MAP_NO_WATER = 0x00000000
LIQUID_MAP_ABOVE_WATER = 0x00000001
LIQUID_MAP_WATER_WALK = 0x00000002
LIQUID_MAP_IN_WATER = 0x00000004
LIQUID_MAP_UNDER_WATER = 0x00000008

MAP_LIQUID_STATUS_SWIMMING = LIQUID_MAP_IN_WATER | LIQUID_MAP_UNDER_WATER
MAP_LIQUID_STATUS_IN_CONTACT = MAP_LIQUID_STATUS_SWIMMING | LIQUID_MAP_WATER_WALK

# AreaTable flags
AREA_FLAG_INSIDE = 0x02000000
AREA_FLAG_OUTSIDE = 0x04000000

# Unit profile for UpdateAllowedPositionZ
DEFAULT_UNIT_PROFILE = {
    "can_swim": True,
    "can_fly": False,
    "hover_height": 0.0,
    "collision_height": DEFAULT_COLLISION_HEIGHT,
    "collision_width": DEFAULT_COLLISION_WIDTH,
}


def _fuzzy_ge32(a: float, b: float) -> bool:
    """G3D fuzzyGe(float): a > b - eps(a)."""
    eps = 1e-5 * (abs(a) + 1.0)
    return a > b - eps


def _fuzzy_lt32(a: float, b: float) -> bool:
    """G3D fuzzyLt(float): a < b + eps(a)."""
    eps = 1e-5 * (abs(a) + 1.0)
    return a < b + eps


@dataclass
class LiquidStatus:
    entry: int = 0
    flags: int = 0
    level: float = INVALID_HEIGHT
    depth_level: float = INVALID_HEIGHT
    status: int = LIQUID_MAP_NO_WATER


@dataclass
class VMAPAreaInfo:
    group_id: int = 0
    adt_id: int = 0
    root_id: int = 0
    mogp_flags: int = 0
    unique_id: int = 0


@dataclass
class VMAPData:
    floor_z: float = VMAP_INVALID_HEIGHT_VALUE
    area: Optional[VMAPAreaInfo] = None
    liquid: Optional[Tuple[int, float]] = None  # (liquid type entry, level)


@dataclass
class TerrainStatus:
    area_id: int = 0
    floor_z: float = VMAP_INVALID_HEIGHT_VALUE
    outdoors: bool = False
    liquid: LiquidStatus = field(default_factory=LiquidStatus)


@dataclass
class AreaInfo:
    ok: bool = False
    mogp_flags: int = 0
    adt_id: int = 0
    root_id: int = 0
    group_id: int = 0
    vmap_z: float = 0.0


class MapResolver:
    """Static server-parity queries for a single map."""

    def __init__(self, map_id: int, dbc_lookup: Optional[Callable] = None,
                 data_paths: Optional[dict] = None):
        self.map_id = map_id
        self.dbc_lookup = dbc_lookup
        self.paths = data_paths or get_data_paths()
        self._map_reader: Optional[MapReader] = None
        self._vmap_tree: Optional[StaticMapTree] = None
        self._vmap_checked = False

        self._area_rows: Optional[Dict[int, dict]] = None
        self._liquid_rows: Optional[Dict[int, dict]] = None
        self._wmo_area_index: Optional[Dict[Tuple[int, int, int], dict]] = None
        self._world_map_area: Optional[Dict[int, Tuple[float, float, float, float]]] = None
        self._map_row = None
        self._map_row_loaded = False
        self._wmo_area_failed = False

    # ---- lazy data sources ----

    def _reader(self) -> MapReader:
        if self._map_reader is None:
            self._map_reader = MapReader(self.paths["maps"])
        return self._map_reader

    def _tree(self) -> Optional[StaticMapTree]:
        if not self._vmap_checked:
            self._vmap_checked = True
            try:
                tree = StaticMapTree(self.map_id, self.paths["vmaps"])
                self._vmap_tree = tree if tree.valid else None
            except Exception:
                self._vmap_tree = None
        return self._vmap_tree

    def _dbc(self, name: str):
        if self.dbc_lookup is None:
            return None
        try:
            return self.dbc_lookup(name)
        except Exception:
            return None

    def _rows(self, name: str):
        reader = self._dbc(name)
        if reader is None:
            return []
        return reader.records

    def _area_row(self, area_id: int):
        if self._area_rows is None:
            self._area_rows = {}
            for row in self._rows("AreaTable"):
                self._area_rows[row.get(0)] = row
        return self._area_rows.get(area_id)

    def _liquid_row(self, liquid_id: int):
        if self._liquid_rows is None:
            self._liquid_rows = {}
            for row in self._rows("LiquidType"):
                self._liquid_rows[row.get(0)] = row
        return self._liquid_rows.get(liquid_id)

    def _wmo_area(self, root_id: int, adt_id: int, group_id: int):
        if self._wmo_area_index is None:
            index: Dict[Tuple[int, int, int], dict] = {}
            for row in self._rows("WMOAreaTable"):
                key = (row.get(1), row.get(2), row.get(3))
                index[key] = row
            self._wmo_area_index = index
        return self._wmo_area_index.get((root_id, adt_id, group_id))

    def _map_row_dbc(self):
        if not self._map_row_loaded:
            self._map_row_loaded = True
            for row in self._rows("Map"):
                if row.get(0) == self.map_id:
                    self._map_row = row
                    break
        return self._map_row

    def map_name(self) -> str:
        row = self._map_row_dbc()
        if row and row.get(5):
            return row[5]
        return ""

    def _linked_zone(self) -> int:
        row = self._map_row_dbc()
        return int(row.get(22) or 0) if row else 0

    def _liquid_flags(self, liquid_type: int) -> int:
        row = self._liquid_row(liquid_type)
        return (1 << (row.get(3) or 0)) if row else 0

    def _world_map_area_index(self):
        if self._world_map_area is not None:
            return self._world_map_area
        self._world_map_area = {}
        path = Path(self.paths["maps"]).parent / "dbc" / "WorldMapArea.dbc"
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            return self._world_map_area
        if len(data) < 20 or data[:4] != b"WDBC":
            return self._world_map_area
        record_count, field_count, record_size = struct.unpack_from("<III", data, 4)
        pos = 20
        for _ in range(record_count):
            if pos + record_size > len(data):
                break
            area_id = struct.unpack_from("<I", data, pos)[0]
            y1, y2, x1, x2 = struct.unpack_from("<4f", data, pos + 16)
            self._world_map_area[area_id] = (x1, x2, y1, y2)
            pos += record_size
        return self._world_map_area

    # ---- ADT grid data ----

    def get_grid_height(self, x: float, y: float) -> float:
        return self._reader().get_height(self.map_id, x, y)

    def get_grid_area(self, x: float, y: float) -> int:
        return self._reader().get_area(self.map_id, x, y)

    def get_height_type(self, x: float, y: float) -> str:
        tile = self._tile(x, y)
        if tile is None or tile.height is None:
            return "none"
        return tile.height.height_type

    def _tile(self, x: float, y: float):
        tx, ty = world_to_map_tile(x, y)
        return self._reader().get_tile(self.map_id, tx, ty)

    def _grid_liquid(self, x: float, y: float, z: float,
                     collision_height: float,
                     req_liquid: Optional[int] = None) -> LiquidStatus:
        """Port of GridTerrainData::GetLiquidData."""
        liquid = LiquidStatus()
        tile = self._tile(x, y)
        if tile is None or tile.liquid is None:
            return liquid
        ld = tile.liquid
        if not (ld.global_flags or ld.flags):
            return liquid

        cx = MAP_RESOLUTION * (32 - x / GRID_SIZE)
        cy = MAP_RESOLUTION * (32 - y / GRID_SIZE)
        x_int = int(cx) & (MAP_RESOLUTION - 1)
        y_int = int(cy) & (MAP_RESOLUTION - 1)
        idx = (x_int >> 3) * 16 + (y_int >> 3)

        liq_type_byte = ld.flags[idx] if ld.flags is not None else ld.global_flags
        entry = ld.entries[idx] if ld.entries is not None else ld.global_entry
        liquid_row = self._liquid_row(entry)
        if liquid_row is not None:
            liq_type_byte &= LIQUID_TYPE_DARK_WATER
            liq_type_idx = liquid_row.get(3) or 0
            if entry < 21:
                area_row = self._area_row(self.get_grid_area(x, y))
                if area_row:
                    override = self._liquid_override(area_row, liq_type_idx)
                    override_row = self._liquid_row(override)
                    if override_row:
                        entry = override
                        liq_type_idx = override_row.get(3) or 0
            liq_type_byte |= 1 << liq_type_idx

        if liq_type_byte != 0 and (req_liquid is None or (req_liquid & liq_type_byte)):
            lx = x_int - ld.off_y
            ly = y_int - ld.off_x
            if 0 <= lx < ld.height and 0 <= ly < ld.width:
                if ld.liquid_map is not None:
                    liquid_level = ld.liquid_map[lx * ld.width + ly]
                else:
                    liquid_level = ld.level
                ground_level = self.get_grid_height(x, y)
                if liquid_level >= ground_level and z >= ground_level - 0.2:
                    liquid.entry = entry
                    liquid.flags = liq_type_byte
                    liquid.level = liquid_level
                    liquid.depth_level = ground_level
                    delta = liquid_level - z
                    if delta > collision_height:
                        liquid.status = LIQUID_MAP_UNDER_WATER
                    elif delta > 0.0:
                        liquid.status = LIQUID_MAP_IN_WATER
                    elif delta > -0.1:
                        liquid.status = LIQUID_MAP_WATER_WALK
                    else:
                        liquid.status = LIQUID_MAP_ABOVE_WATER
        return liquid

    @staticmethod
    def _liquid_override(area_row: dict, liq_type_idx: int) -> int:
        if liq_type_idx < 0 or liq_type_idx > 3:
            return 0
        return area_row.get(29 + liq_type_idx) or 0

    # ---- vmap data ----

    def get_vmap_data(self, x: float, y: float, z: float,
                      req_liquid: Optional[int] = None) -> Optional[VMAPData]:
        """Port of StaticVMapCollisionData::GetAreaAndLiquidData."""
        tree = self._tree()
        if tree is None:
            return None
        tree.ensure_tiles_for_point(x, y)
        pos = to_internal(x, y, z)
        info = tree.get_location_info(pos)
        if not info:
            return None

        data = VMAPData(floor_z=info["ground_z"])
        hit_group = info.get("hit_group")
        hit_instance = info.get("hit_instance")
        if hit_group is not None and hit_instance is not None:
            if req_liquid is None or (self._liquid_flags(hit_group.liquid_type) & req_liquid):
                level = hit_instance.get_liquid_level(pos, info)
                if level is not None:
                    data.liquid = (hit_group.liquid_type, level)
            data.area = VMAPAreaInfo(
                group_id=hit_group.group_wmo_id,
                adt_id=hit_instance.adt_id,
                root_id=info["root_id"],
                mogp_flags=hit_group.mogp_flags,
                unique_id=hit_instance.id,
            )
        return data

    # ---- heights ----

    def get_height(self, x: float, y: float, z: float,
                   check_vmap: bool = True,
                   max_search_dist: float = DEFAULT_HEIGHT_SEARCH) -> float:
        """Port of Map::GetHeight (static part)."""
        map_height = VMAP_INVALID_HEIGHT_VALUE
        grid_height = self.get_grid_height(x, y)
        if _fuzzy_ge32(z, grid_height - GROUND_HEIGHT_TOLERANCE):
            map_height = grid_height

        vmap_height = VMAP_INVALID_HEIGHT_VALUE
        if check_vmap:
            tree = self._tree()
            if tree is not None:
                h = tree.get_height(x, y, z, max_search_dist)
                if not math.isinf(h):
                    vmap_height = h

        if vmap_height > INVALID_HEIGHT:
            if map_height > INVALID_HEIGHT:
                if vmap_height > map_height or \
                        abs(map_height - z) > abs(vmap_height - z):
                    return vmap_height
                return map_height
            return vmap_height
        return map_height

    def get_ground_height(self, x: float, y: float) -> float:
        """WorldObject::GetMapHeight(x, y, MAX_HEIGHT) - the raw terrain Z."""
        return self.get_height(x, y, MAX_HEIGHT, True, DEFAULT_HEIGHT_SEARCH)

    def get_floor_height(self, x: float, y: float, z: float,
                         collision_height: float = DEFAULT_COLLISION_HEIGHT) -> float:
        """WorldObject::GetMapHeight(x, y, z) - the vmap-aware floor Z."""
        search_z = z + max(collision_height, Z_OFFSET_FIND_HEIGHT)
        return self.get_height(x, y, search_z, True, DEFAULT_HEIGHT_SEARCH)

    def get_water_or_ground_level(self, x: float, y: float, z: float,
                                  collision_height: float = DEFAULT_COLLISION_HEIGHT):
        """Port of Map::GetWaterOrGroundLevel. Returns (level, ground_z)."""
        ground_z = self.get_height(x, y, z + Z_OFFSET_FIND_HEIGHT, True, 50.0)
        liquid = self.get_liquid_data(x, y, ground_z, collision_height)
        if liquid.status == LIQUID_MAP_ABOVE_WATER:
            return max(liquid.level, ground_z), ground_z
        if liquid.status == LIQUID_MAP_NO_WATER:
            return ground_z, ground_z
        return liquid.level, ground_z

    # ---- areas ----

    def get_area_info(self, x: float, y: float, z: float) -> AreaInfo:
        """Port of Map::GetAreaInfo (static tree only)."""
        info = AreaInfo(ok=False, adt_id=0, root_id=0, group_id=0, mogp_flags=0,
                        vmap_z=z)
        data = self.get_vmap_data(x, y, z)
        if data is None or data.area is None:
            return info
        info.ok = True
        info.mogp_flags = data.area.mogp_flags
        info.adt_id = data.area.adt_id
        info.root_id = data.area.root_id
        info.group_id = data.area.group_id
        info.vmap_z = data.floor_z

        grid_height = self.get_grid_height(x, y)
        if z + 2.0 > grid_height and grid_height > info.vmap_z:
            info.ok = False
        return info

    def get_area_id(self, x: float, y: float, z: float) -> int:
        """Port of Map::GetAreaId."""
        area_info = self.get_area_info(x, y, z)
        grid_area_id = self.get_grid_area(x, y)
        grid_height = self.get_grid_height(x, y)

        area_id = 0
        if area_info.ok and _fuzzy_ge32(z, area_info.vmap_z - GROUND_HEIGHT_TOLERANCE) \
                and (_fuzzy_lt32(z, grid_height - GROUND_HEIGHT_TOLERANCE)
                     or area_info.vmap_z > grid_height):
            wmo = self._wmo_area(area_info.root_id, area_info.adt_id,
                                 area_info.group_id)
            if wmo:
                area_id = wmo.get(10) or 0
            if not area_id:
                area_id = grid_area_id
        else:
            area_id = grid_area_id

        if not area_id:
            area_id = self._linked_zone()
        return area_id

    def get_zone_and_area_id(self, x: float, y: float, z: float):
        """Port of Map::GetZoneAndAreaId. Returns (zone_id, area_id)."""
        area_id = self.get_area_id(x, y, z)
        zone_id = area_id
        area_row = self._area_row(area_id)
        if area_row and area_row.get(2):
            zone_id = area_row[2]
        return zone_id, area_id

    def area_name(self, area_id: int) -> str:
        row = self._area_row(area_id)
        if row and row.get(11):
            return row[11]
        return ""

    def map2zone_coordinates(self, x: float, y: float, zone_id: int):
        """Port of Map2ZoneCoordinates. Returns (zone_x, zone_y)."""
        ma = self._world_map_area_index().get(zone_id)
        if ma is None:
            return x, y
        x1, x2, y1, y2 = ma
        if x2 == x1 or y2 == y1:
            return x, y
        zx = (x - x1) / ((x2 - x1) / 100)
        zy = (y - y1) / ((y2 - y1) / 100)
        return zy, zx

    # ---- liquids ----

    def get_liquid_data(self, x: float, y: float, z: float,
                        collision_height: float = DEFAULT_COLLISION_HEIGHT,
                        req_liquid: Optional[int] = None) -> LiquidStatus:
        """Port of Map::GetLiquidData (static part)."""
        liquid = LiquidStatus()
        vmap_data = self.get_vmap_data(x, y, z, req_liquid)
        use_grid_liquid = True
        vmap_floor = VMAP_INVALID_HEIGHT_VALUE
        if vmap_data is not None:
            vmap_floor = vmap_data.floor_z
            if vmap_data.liquid is not None:
                if vmap_data.area is not None:
                    use_grid_liquid = (vmap_data.area.mogp_flags & 0x2000) == 0
                liquid_type, level = vmap_data.liquid
                if level > vmap_floor and _fuzzy_ge32(
                        z, vmap_floor - GROUND_HEIGHT_TOLERANCE):
                    if self.map_id == 530 and liquid_type == 2:
                        liquid_type = 15
                    flag_type = self._liquid_row(liquid_type)
                    liquid_flag_type = (flag_type.get(3) or 0) if flag_type else 0

                    if liquid_type and liquid_type < 21:
                        area_row = self._area_row(self.get_area_id(x, y, z))
                        override = self._liquid_override(area_row, liquid_flag_type) if area_row else 0
                        if not override and area_row and area_row.get(2):
                            zone_row = self._area_row(area_row[2])
                            override = self._liquid_override(zone_row, liquid_flag_type) if zone_row else 0
                        override_row = self._liquid_row(override)
                        if override_row:
                            liquid_type = override
                            liquid_flag_type = override_row.get(3) or 0

                    liquid.entry = liquid_type
                    liquid.flags = 1 << liquid_flag_type
                    liquid.level = vmap_data.liquid[1]
                    liquid.depth_level = vmap_floor
                    delta = liquid.level - z
                    if delta > collision_height:
                        liquid.status = LIQUID_MAP_UNDER_WATER
                    elif delta > 0.0:
                        liquid.status = LIQUID_MAP_IN_WATER
                    elif delta > -0.1:
                        liquid.status = LIQUID_MAP_WATER_WALK
                    else:
                        liquid.status = LIQUID_MAP_ABOVE_WATER

        if use_grid_liquid:
            grid = self._grid_liquid(x, y, z, collision_height, req_liquid)
            if grid.status != LIQUID_MAP_NO_WATER and \
                    (vmap_data is None or grid.level > vmap_floor):
                liquid_entry = grid.entry
                if self.map_id == 530 and liquid_entry == 2:
                    liquid_entry = 15
                liquid = grid
                liquid.entry = liquid_entry
        return liquid

    def is_in_water(self, x: float, y: float, z: float,
                    collision_height: float = DEFAULT_COLLISION_HEIGHT) -> bool:
        liquid = self.get_liquid_data(x, y, z, collision_height)
        return (liquid.status & MAP_LIQUID_STATUS_SWIMMING) != 0

    # ---- full terrain status ----

    def get_full_terrain_status(self, x: float, y: float, z: float,
                                collision_height: float = DEFAULT_COLLISION_HEIGHT,
                                req_liquid: Optional[int] = None) -> TerrainStatus:
        """Port of Map::GetFullTerrainStatusForPosition (static part)."""
        status = TerrainStatus()
        grid_area_id = self.get_grid_area(x, y)
        grid_map_height = self.get_grid_height(x, y)

        vmap_data = self.get_vmap_data(x, y, z, req_liquid)
        wmo_data = None

        use_grid_liquid = True
        status.floor_z = VMAP_INVALID_HEIGHT_VALUE
        if grid_map_height > INVALID_HEIGHT and \
                _fuzzy_ge32(z, grid_map_height - GROUND_HEIGHT_TOLERANCE):
            status.floor_z = grid_map_height

        if vmap_data is not None and vmap_data.floor_z > VMAP_INVALID_HEIGHT and \
                _fuzzy_ge32(z, vmap_data.floor_z - GROUND_HEIGHT_TOLERANCE) and \
                (_fuzzy_lt32(z, grid_map_height - GROUND_HEIGHT_TOLERANCE)
                 or vmap_data.floor_z > grid_map_height):
            status.floor_z = vmap_data.floor_z
            wmo_data = vmap_data

        if wmo_data is not None:
            if wmo_data.area is not None:
                wmo_entry = self._wmo_area(wmo_data.area.root_id,
                                           wmo_data.area.adt_id,
                                           wmo_data.area.group_id)
                status.outdoors = (wmo_data.area.mogp_flags & 0x8) != 0
                if wmo_entry:
                    status.area_id = wmo_entry.get(10) or 0
                    flags = wmo_entry.get(9) or 0
                    if flags & 4:
                        status.outdoors = True
                    elif flags & 2:
                        status.outdoors = False
                if not status.area_id:
                    status.area_id = grid_area_id
                use_grid_liquid = (wmo_data.area.mogp_flags & 0x2000) == 0
        else:
            status.outdoors = True
            status.area_id = grid_area_id
            area_row = self._area_row(status.area_id)
            if area_row:
                status.outdoors = (area_row.get(4, 0) & (AREA_FLAG_INSIDE | AREA_FLAG_OUTSIDE)) != AREA_FLAG_INSIDE

        if not status.area_id:
            status.area_id = self._linked_zone()
        area_row = self._area_row(status.area_id)

        # WMO liquid
        if wmo_data is not None and wmo_data.liquid is not None and \
                wmo_data.liquid[1] > wmo_data.floor_z:
            liquid_type = wmo_data.liquid[0]
            if self.map_id == 530 and liquid_type == 2:
                liquid_type = 15
            liquid_flag_type = 0
            liquid_row = self._liquid_row(liquid_type)
            if liquid_row:
                liquid_flag_type = liquid_row.get(3) or 0
            if liquid_type and liquid_type < 21 and area_row:
                override = self._liquid_override(area_row, liquid_flag_type)
                if not override and area_row.get(2):
                    zone_row = self._area_row(area_row[2])
                    if zone_row:
                        override = self._liquid_override(zone_row, liquid_flag_type)
                override_row = self._liquid_row(override)
                if override_row:
                    liquid_type = override
                    liquid_flag_type = override_row.get(3) or 0

            status.liquid.entry = liquid_type
            status.liquid.flags = 1 << liquid_flag_type
            status.liquid.level = wmo_data.liquid[1]
            status.liquid.depth_level = wmo_data.floor_z
            delta = wmo_data.liquid[1] - z
            if delta > collision_height:
                status.liquid.status = LIQUID_MAP_UNDER_WATER
            elif delta > 0.0:
                status.liquid.status = LIQUID_MAP_IN_WATER
            elif delta > -0.1:
                status.liquid.status = LIQUID_MAP_WATER_WALK
            else:
                status.liquid.status = LIQUID_MAP_ABOVE_WATER

        # grid liquid
        if use_grid_liquid:
            grid = self._grid_liquid(x, y, z, collision_height, req_liquid)
            if grid.status != LIQUID_MAP_NO_WATER and \
                    (wmo_data is None or grid.level > wmo_data.floor_z):
                entry = grid.entry
                if self.map_id == 530 and entry == 2:
                    entry = 15
                status.liquid = grid
                status.liquid.entry = entry
        return status

    # ---- line of sight / hit position ----

    def line_of_sight(self, x1: float, y1: float, z1: float,
                      x2: float, y2: float, z2: float,
                      ignore_m2: bool = False) -> bool:
        """StaticVMapCollisionData::isInLineOfSight (no dynamic objects)."""
        tree = self._tree()
        if tree is None:
            return True
        tree.ensure_tiles_for_point(x1, y1)
        tree.ensure_tiles_for_point(x2, y2)
        p1 = to_internal(x1, y1, z1)
        p2 = to_internal(x2, y2, z2)
        if p1 == p2:
            return True
        flags = IGNORE_M2 if ignore_m2 else IGNORE_NOTHING
        return tree.is_in_line_of_sight(p1, p2, flags)

    def get_object_hit_pos(self, x1: float, y1: float, z1: float,
                           x2: float, y2: float, z2: float,
                           modify_dist: float = 0.0):
        """StaticVMapCollisionData::GetObjectHitPos (no dynamic objects).

        Returns (hit, x, y, z); on a miss the end position is returned.
        """
        tree = self._tree()
        if tree is None:
            return False, x2, y2, z2
        tree.ensure_tiles_for_point(x1, y1)
        tree.ensure_tiles_for_point(x2, y2)
        p1 = to_internal(x1, y1, z1)
        p2 = to_internal(x2, y2, z2)
        hit, result = tree.get_object_hit_pos(p1, p2, modify_dist)
        world = to_world(result[0], result[1], result[2])
        return hit, world[0], world[1], world[2]

    # ---- unit positioning ----

    def update_allowed_position_z(self, x: float, y: float, z: float,
                                  profile: Optional[dict] = None) -> float:
        """Port of WorldObject::UpdateAllowedPositionZ for a unit."""
        profile = profile or DEFAULT_UNIT_PROFILE
        if not profile.get("can_fly", False):
            can_swim = profile.get("can_swim", True)
            collision_height = profile.get("collision_height", DEFAULT_COLLISION_HEIGHT)
            hover = profile.get("hover_height", 0.0)
            ground_z = z
            if can_swim:
                max_z, ground_z = self.get_water_or_ground_level(
                    x, y, z, collision_height)
            else:
                max_z = ground_z = self.get_height(x, y, z)

            if max_z > INVALID_HEIGHT:
                if can_swim and self.is_in_water(
                        x, y, max_z - Z_OFFSET_FIND_HEIGHT, collision_height):
                    max_z = max(max_z - _get_min_height_in_water(profile), ground_z)
                else:
                    max_z += hover
                    ground_z += hover
                if z > max_z:
                    z = max_z
                elif z < ground_z:
                    z = ground_z
        return z


def _get_min_height_in_water(profile: dict) -> float:
    height = profile.get("collision_height", DEFAULT_COLLISION_HEIGHT)
    width = profile.get("collision_width", DEFAULT_COLLISION_WIDTH)
    if width <= 0:
        return height
    area = math.pi * (width / 2.0) ** 2
    weight = area * height * 1040.0
    height_out_of_water = (weight / (area * 10202.0)) * 4.0
    height_in_water = height - height_out_of_water
    return height_in_water if height > height_in_water else height - height / 3.0
