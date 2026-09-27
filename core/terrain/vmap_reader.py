"""Metadata reader for VMAP_4.8 .vmtree/.vmtile files.

The tile file layout is ``magic[8] + u32 numSpawns`` followed by
``numSpawns`` × (ModelSpawn + u32 treeRef) records.  The tree file is
``magic[8] + u8 tiled + "NODE" + BIH`` (+ ``"GOBJ"`` and, for non-tiled
maps, one global ModelSpawn).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

from core.terrain.vmap.bih import BIH
from core.terrain.vmap.tree import read_model_spawn
from core.terrain.vmap.world_model import MOD_M2, VMAP_MAGIC


class VMapTileInfo(NamedTuple):
    """Metadata about a vmap tile or tree."""
    map_id: int
    tile_x: int
    tile_y: int
    is_tiled: bool
    model_count: int       # distinct model file names
    spawn_count: int       # spawn records in the file
    gobject_count: int
    file_size: int
    has_bih_tree: bool
    wmo_spawns: int = 0
    models: Tuple[str, ...] = ()


class VMapReader:
    """Reads .vmtree/.vmtile metadata (no geometry parsing)."""

    def __init__(self, vmaps_path):
        self.vmaps_path = Path(vmaps_path)
        self._cache: Dict[Tuple[int, int, int], VMapTileInfo] = {}

    @staticmethod
    def tile_filename(map_id: int, tile_x: int, tile_y: int) -> str:
        # StaticMapTree::getTileFileName writes tileY first.
        return f"{map_id:03d}_{tile_y:02d}_{tile_x:02d}.vmtile"

    @staticmethod
    def tree_filename(map_id: int) -> str:
        return f"{map_id:03d}.vmtree"

    def get_tile_info(self, map_id: int, tile_x: int, tile_y: int) -> Optional[VMapTileInfo]:
        """tile_x/tile_y are world grid coords (file is written tileY_tileX)."""
        cache_key = (map_id, tile_x, tile_y)
        if cache_key in self._cache:
            return self._cache[cache_key]
        path = self.vmaps_path / self.tile_filename(map_id, tile_x, tile_y)
        if not path.exists():
            return None
        info = self._parse_tile(path, map_id, tile_x, tile_y)
        if info:
            self._cache[cache_key] = info
        return info

    def get_tree_info(self, map_id: int) -> Optional[VMapTileInfo]:
        path = self.vmaps_path / self.tree_filename(map_id)
        if not path.exists():
            return None
        return self._parse_tree(path, map_id)

    def list_tiles(self, map_id: int) -> List[str]:
        if not self.vmaps_path.exists():
            return []
        prefix = f"{map_id:03d}_"
        return sorted(
            f.name for f in self.vmaps_path.iterdir()
            if f.name.startswith(prefix) and f.suffix == ".vmtile"
        )

    # ---- parsing ----

    def _parse_tile(self, path: Path, map_id: int,
                    tile_x: int, tile_y: int) -> Optional[VMapTileInfo]:
        try:
            data = path.read_bytes()
        except OSError:
            return None
        if len(data) < 12 or data[:8] != VMAP_MAGIC:
            return None
        (num_spawns,) = struct.unpack_from("<I", data, 8)
        pos = 12
        names: Dict[str, int] = {}
        wmo_spawns = 0
        spawn_count = 0
        try:
            for _ in range(num_spawns):
                spawn, pos = read_model_spawn(data, pos)
                pos += 4  # tree reference
                names[spawn.name] = names.get(spawn.name, 0) + 1
                if not (spawn.flags & MOD_M2):
                    wmo_spawns += 1
                spawn_count += 1
        except (ValueError, struct.error, IndexError):
            pass
        return VMapTileInfo(
            map_id=map_id,
            tile_x=tile_x,
            tile_y=tile_y,
            is_tiled=True,
            model_count=len(names),
            spawn_count=spawn_count,
            gobject_count=0,
            file_size=len(data),
            has_bih_tree=False,
            wmo_spawns=wmo_spawns,
            models=tuple(sorted(names)[:20]),
        )

    def _parse_tree(self, path: Path, map_id: int) -> Optional[VMapTileInfo]:
        try:
            data = path.read_bytes()
        except OSError:
            return None
        if len(data) < 13 or data[:8] != VMAP_MAGIC:
            return None
        tiled = data[8] != 0
        pos = 9
        has_bih = data[pos:pos + 4] == b"NODE"
        prim_count = 0
        if has_bih:
            bih = BIH()
            try:
                pos = bih.read_from_file(data, pos + 4)
                prim_count = bih.prim_count
            except struct.error:
                prim_count = 0
        has_gobj = data[pos:pos + 4] == b"GOBJ"
        return VMapTileInfo(
            map_id=map_id,
            tile_x=0,
            tile_y=0,
            is_tiled=tiled,
            model_count=prim_count,
            spawn_count=prim_count,
            gobject_count=1 if has_gobj else 0,
            file_size=len(data),
            has_bih_tree=has_bih,
        )
