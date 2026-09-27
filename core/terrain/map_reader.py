"""Binary parser for AzerothCore .map terrain files (MAPS v9).

The indexing mirrors ``GridTerrainData`` exactly: heights use the v9
(129x129) grid plus the v8 (128x128) centre grid with the server's triangle
interpolation, areas use the flat 16x16 map indexed ``x*16+y``, liquid uses
entry/flag grids with the server cell math, and holes are honoured.
"""

import struct
from pathlib import Path
from typing import List, Optional

from core.terrain.coords import (
    MAP_TILE_VERTS_V9,
    world_to_map_tile,
)


# Magic constants
MAP_MAGIC = b"MAPS"
MAP_VERSION = 9
AREA_MAGIC = b"AREA"
HEIGHT_MAGIC = b"MHGT"
LIQUID_MAGIC = b"MLIQ"

# Header flags
MAP_AREA_NO_AREA = 0x0001
MAP_HEIGHT_NO_HEIGHT = 0x0001
MAP_HEIGHT_AS_INT16 = 0x0002
MAP_HEIGHT_AS_INT8 = 0x0004
MAP_HEIGHT_HAS_FLIGHT_BOUNDS = 0x0008
MAP_LIQUID_NO_TYPE = 0x0001
MAP_LIQUID_NO_HEIGHT = 0x0002

# Liquid types
LIQUID_TYPE_NO_WATER = 0x00
LIQUID_TYPE_WATER = 0x01
LIQUID_TYPE_OCEAN = 0x02
LIQUID_TYPE_MAGMA = 0x04
LIQUID_TYPE_SLIME = 0x08
LIQUID_TYPE_DARK_WATER = 0x10

# Liquid status flags
LIQUID_STATUS_NO_WATER = 0x00
LIQUID_STATUS_ABOVE_WATER = 0x01
LIQUID_STATUS_WATER_WALK = 0x02
LIQUID_STATUS_IN_WATER = 0x04
LIQUID_STATUS_UNDER_WATER = 0x08

INVALID_HEIGHT = -100000.0
MAP_RESOLUTION = 128
GRID_SIZE = 533.3333
GROUND_HEIGHT_TOLERANCE = 0.05

_HOLETAB_H = (0x1111, 0x2222, 0x4444, 0x8888)
_HOLETAB_V = (0x000F, 0x00F0, 0x0F00, 0xF000)


class AreaData:
    __slots__ = ("grid_area", "area_map")

    def __init__(self, grid_area: int, area_map: List[int]):
        self.grid_area = grid_area
        self.area_map = area_map  # flat 256, file order


class HeightData:
    __slots__ = ("grid_height", "grid_max_height", "height_type",
                 "v9", "v8", "multiplier")

    def __init__(self, grid_height: float, grid_max_height: float,
                 height_type: str, v9: list, v8: list, multiplier: float):
        self.grid_height = grid_height
        self.grid_max_height = grid_max_height
        self.height_type = height_type
        self.v9 = v9
        self.v8 = v8
        self.multiplier = multiplier


class LiquidData:
    __slots__ = ("global_entry", "global_flags", "off_x", "off_y",
                 "width", "height", "level", "entries", "flags", "liquid_map")

    def __init__(self) -> None:
        self.global_entry = 0
        self.global_flags = 0
        self.off_x = 0
        self.off_y = 0
        self.width = 0
        self.height = 0
        self.level = INVALID_HEIGHT
        self.entries: Optional[List[int]] = None  # 256
        self.flags: Optional[List[int]] = None    # 256
        self.liquid_map: Optional[List[float]] = None  # width*height


class MapTile:
    __slots__ = ("map_id", "tile_x", "tile_y", "area", "height", "liquid",
                 "holes")

    def __init__(self, map_id: int, tile_x: int, tile_y: int):
        self.map_id = map_id
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.area: Optional[AreaData] = None
        self.height: Optional[HeightData] = None
        self.liquid: Optional[LiquidData] = None
        self.holes: Optional[List[int]] = None

    def is_hole(self, row: int, col: int) -> bool:
        """Port of GridTerrainData::isHole (row=x_int, col=y_int)."""
        if not self.holes:
            return False
        cell_row = row // 8
        cell_col = col // 8
        hole_row = (row % 8) // 2
        hole_col = (col - cell_col * 8) // 2
        hole = self.holes[cell_row * 16 + cell_col]
        return (hole & _HOLETAB_H[hole_col] & _HOLETAB_V[hole_row]) != 0

    def get_height(self, x: float, y: float) -> float:
        """World-space height (port of GridTerrainData::getHeight)."""
        hd = self.height
        if hd is None:
            return INVALID_HEIGHT
        if hd.height_type == "flat":
            return hd.grid_height

        rx = MAP_RESOLUTION * (32 - x / GRID_SIZE)
        ry = MAP_RESOLUTION * (32 - y / GRID_SIZE)
        x_int = int(rx)
        y_int = int(ry)
        fx = rx - x_int
        fy = ry - y_int
        x_int &= MAP_RESOLUTION - 1
        y_int &= MAP_RESOLUTION - 1

        if self.is_hole(x_int, y_int):
            return INVALID_HEIGHT

        if hd.height_type == "uint8":
            return self._height_uint8(hd, x_int, y_int, fx, fy)
        if hd.height_type == "uint16":
            return self._height_uint16(hd, x_int, y_int, fx, fy)
        return self._height_float(hd, x_int, y_int, fx, fy)

    @staticmethod
    def _height_float(hd: HeightData, x_int: int, y_int: int,
                      x: float, y: float) -> float:
        v9 = hd.v9
        v8 = hd.v8
        if x + y < 1:
            if x > y:
                h1 = v9[x_int * 129 + y_int]
                h2 = v9[(x_int + 1) * 129 + y_int]
                h5 = 2 * v8[x_int * 128 + y_int]
                a = h2 - h1
                b = h5 - h1 - h2
                c = h1
            else:
                h1 = v9[x_int * 129 + y_int]
                h3 = v9[x_int * 129 + y_int + 1]
                h5 = 2 * v8[x_int * 128 + y_int]
                a = h5 - h1 - h3
                b = h3 - h1
                c = h1
        else:
            if x > y:
                h2 = v9[(x_int + 1) * 129 + y_int]
                h4 = v9[(x_int + 1) * 129 + y_int + 1]
                h5 = 2 * v8[x_int * 128 + y_int]
                a = h2 + h4 - h5
                b = h4 - h2
                c = h5 - h4
            else:
                h3 = v9[x_int * 129 + y_int + 1]
                h4 = v9[(x_int + 1) * 129 + y_int + 1]
                h5 = 2 * v8[x_int * 128 + y_int]
                a = h4 - h3
                b = h3 + h4 - h5
                c = h5 - h4
        return a * x + b * y + c

    @staticmethod
    def _height_uint16(hd: HeightData, x_int: int, y_int: int,
                       x: float, y: float) -> float:
        v9 = hd.v9
        v8 = hd.v8
        i = x_int * 128 + x_int + y_int
        if x + y < 1:
            if x > y:
                h1 = v9[i]
                h2 = v9[i + 129]
                h5 = 2 * v8[x_int * 128 + y_int]
                a = h2 - h1
                b = h5 - h1 - h2
                c = h1
            else:
                h1 = v9[i]
                h3 = v9[i + 1]
                h5 = 2 * v8[x_int * 128 + y_int]
                a = h5 - h1 - h3
                b = h3 - h1
                c = h1
        else:
            if x > y:
                h2 = v9[i + 129]
                h4 = v9[i + 130]
                h5 = 2 * v8[x_int * 128 + y_int]
                a = h2 + h4 - h5
                b = h4 - h2
                c = h5 - h4
            else:
                h3 = v9[i + 1]
                h4 = v9[i + 130]
                h5 = 2 * v8[x_int * 128 + y_int]
                a = h4 - h3
                b = h3 + h4 - h5
                c = h5 - h4
        return ((a * x) + (b * y) + c) * hd.multiplier + hd.grid_height

    @staticmethod
    def _height_uint8(hd: HeightData, x_int: int, y_int: int,
                      x: float, y: float) -> float:
        v9 = hd.v9
        v8 = hd.v8
        i = x_int * 128 + x_int + y_int
        if x + y < 1:
            if x > y:
                h1 = v9[i]
                h2 = v9[i + 129]
                h5 = 2 * v8[x_int * 128 + y_int]
                a = h2 - h1
                b = h5 - h1 - h2
                c = h1
            else:
                h1 = v9[i]
                h3 = v9[i + 1]
                h5 = 2 * v8[x_int * 128 + y_int]
                a = h5 - h1 - h3
                b = h3 - h1
                c = h1
        else:
            if x > y:
                h2 = v9[i + 129]
                h4 = v9[i + 130]
                h5 = 2 * v8[x_int * 128 + y_int]
                a = h2 + h4 - h5
                b = h4 - h2
                c = h5 - h4
            else:
                h3 = v9[i + 1]
                h4 = v9[i + 130]
                h5 = 2 * v8[x_int * 128 + y_int]
                a = h4 - h3
                b = h3 + h4 - h5
                c = h5 - h4
        return ((a * x) + (b * y) + c) * hd.multiplier + hd.grid_height

    def get_area(self, x: float, y: float) -> int:
        """World-space area id (port of GridTerrainData::getArea)."""
        if not self.area:
            return 0
        if not self.area.area_map:
            return self.area.grid_area
        rx = 16 * (32 - x / GRID_SIZE)
        ry = 16 * (32 - y / GRID_SIZE)
        lx = int(rx) & 15
        ly = int(ry) & 15
        return self.area.area_map[lx * 16 + ly]


class MapReader:
    """Reads and caches .map terrain files."""

    def __init__(self, maps_path: Path):
        self.maps_path = Path(maps_path)
        self._cache = {}

    def _get_tile_path(self, map_id: int, tile_x: int, tile_y: int) -> Path:
        from core.terrain.coords import map_tile_filename
        return self.maps_path / map_tile_filename(map_id, tile_x, tile_y)

    def get_tile(self, map_id: int, tile_x: int, tile_y: int) -> Optional[MapTile]:
        """Get or load a map tile. Returns None if the file doesn't exist."""
        cache_key = (map_id, tile_x, tile_y)
        if cache_key in self._cache:
            return self._cache[cache_key]

        tile_path = self._get_tile_path(map_id, tile_x, tile_y)
        if not tile_path.exists():
            return None

        tile = MapTile(map_id, tile_x, tile_y)
        try:
            self._parse_file(tile_path, tile)
        except Exception as e:
            raise RuntimeError(f"Failed to parse {tile_path.name}: {e}") from e

        self._cache[cache_key] = tile
        return tile

    def get_height(self, map_id: int, world_x: float, world_y: float) -> float:
        """Terrain height at world coordinates (INVALID_HEIGHT when absent)."""
        tile_x, tile_y = world_to_map_tile(world_x, world_y)
        tile = self.get_tile(map_id, tile_x, tile_y)
        if not tile:
            return INVALID_HEIGHT
        return tile.get_height(world_x, world_y)

    def get_area(self, map_id: int, world_x: float, world_y: float) -> int:
        tile_x, tile_y = world_to_map_tile(world_x, world_y)
        tile = self.get_tile(map_id, tile_x, tile_y)
        if not tile:
            return 0
        return tile.get_area(world_x, world_y)

    # ---- parsing ----

    def _parse_file(self, path: Path, tile: MapTile) -> None:
        with open(path, "rb") as f:
            data = f.read()

        if len(data) < 44 or data[0:4] != MAP_MAGIC:
            raise ValueError("Invalid map magic")
        version, build = struct.unpack_from("<II", data, 4)
        if version != MAP_VERSION:
            raise ValueError(f"Unsupported map version: {version}")
        (area_offset, area_size, height_offset, height_size,
         liquid_offset, liquid_size, holes_offset, holes_size
         ) = struct.unpack_from("<IIIIIIII", data, 12)

        if area_offset > 0 and area_size > 0:
            tile.area = self._parse_area(data, area_offset)
        if height_offset > 0 and height_size > 0:
            tile.height = self._parse_height(data, height_offset)
        if liquid_offset > 0 and liquid_size > 0:
            tile.liquid = self._parse_liquid(data, liquid_offset)
        if holes_offset > 0 and holes_size > 0 and holes_size >= 512:
            tile.holes = list(struct.unpack_from("<256H", data, holes_offset))

    @staticmethod
    def _parse_area(data: bytes, offset: int) -> AreaData:
        if data[offset:offset + 4] != AREA_MAGIC:
            raise ValueError("Invalid area magic")
        flags, grid_area = struct.unpack_from("<Hh", data, offset + 4)
        if flags & MAP_AREA_NO_AREA:
            return AreaData(grid_area=grid_area, area_map=[])
        area_map = list(struct.unpack_from("<256H", data, offset + 8))
        return AreaData(grid_area=grid_area, area_map=area_map)

    @staticmethod
    def _parse_height(data: bytes, offset: int) -> HeightData:
        if data[offset:offset + 4] != HEIGHT_MAGIC:
            raise ValueError("Invalid height magic")
        flags, grid_height, grid_max_height = struct.unpack_from(
            "<Iff", data, offset + 4
        )
        pos = offset + 16

        if flags & MAP_HEIGHT_NO_HEIGHT:
            return HeightData(grid_height, grid_max_height, "flat", [], [], 1.0)

        n9 = MAP_TILE_VERTS_V9  # 129
        n8 = 128
        if flags & MAP_HEIGHT_AS_INT8:
            v9 = list(struct.unpack_from(f"<{n9 * n9}B", data, pos))
            pos += n9 * n9
            v8 = list(struct.unpack_from(f"<{n8 * n8}B", data, pos))
            multiplier = (grid_max_height - grid_height) / 255.0
            return HeightData(grid_height, grid_max_height, "uint8", v9, v8,
                              multiplier)
        if flags & MAP_HEIGHT_AS_INT16:
            v9 = list(struct.unpack_from(f"<{n9 * n9}H", data, pos))
            pos += n9 * n9 * 2
            v8 = list(struct.unpack_from(f"<{n8 * n8}H", data, pos))
            multiplier = (grid_max_height - grid_height) / 65535.0
            return HeightData(grid_height, grid_max_height, "uint16", v9, v8,
                              multiplier)
        v9 = list(struct.unpack_from(f"<{n9 * n9}f", data, pos))
        pos += n9 * n9 * 4
        v8 = list(struct.unpack_from(f"<{n8 * n8}f", data, pos))
        return HeightData(grid_height, grid_max_height, "float", v9, v8, 1.0)

    @staticmethod
    def _parse_liquid(data: bytes, offset: int) -> LiquidData:
        if data[offset:offset + 4] != LIQUID_MAGIC:
            raise ValueError("Invalid liquid magic")
        (flags, liquid_flag, liquid_type, off_x, off_y,
         width, height, liquid_level
         ) = struct.unpack_from("<BBhBBBBf", data, offset + 4)

        liquid = LiquidData()
        liquid.global_entry = liquid_type
        liquid.global_flags = liquid_flag
        liquid.off_x = off_x
        liquid.off_y = off_y
        liquid.width = width
        liquid.height = height
        liquid.level = liquid_level

        pos = offset + 16
        if not (flags & MAP_LIQUID_NO_TYPE):
            liquid.entries = list(struct.unpack_from("<256H", data, pos))
            pos += 512
            liquid.flags = list(struct.unpack_from("<256B", data, pos))
            pos += 256
        if not (flags & MAP_LIQUID_NO_HEIGHT):
            count = width * height
            liquid.liquid_map = list(struct.unpack_from(f"<{count}f", data, pos))
        return liquid
