"""Bounding Interval Hierarchy reader/traverser (port of the vendored BIH)."""

from __future__ import annotations

import struct
from typing import Callable, List, Tuple

from core.terrain.vmap.math3d import (
    Vec3,
    float_to_raw_bits,
    fuzzy_ne32,
    int_bits_to_float,
)

MAX_STACK_SIZE = 64


class BIH:
    """Bounding Interval Hierarchy used by vmap tree/model files."""

    __slots__ = ("bounds_lo", "bounds_hi", "tree", "objects")

    def __init__(self) -> None:
        self.bounds_lo: Vec3 = (0.0, 0.0, 0.0)
        self.bounds_hi: Vec3 = (0.0, 0.0, 0.0)
        self.tree: List[int] = [3 << 30, 0, 0]
        self.objects: List[int] = []

    @property
    def prim_count(self) -> int:
        return len(self.objects)

    def read_from_file(self, data: bytes, offset: int) -> int:
        """Parse from ``data`` at ``offset``; returns the new offset."""
        lo = struct.unpack_from("<3f", data, offset)
        hi = struct.unpack_from("<3f", data, offset + 12)
        (tree_size,) = struct.unpack_from("<I", data, offset + 24)
        pos = offset + 28
        tree = list(struct.unpack_from(f"<{tree_size}I", data, pos)) if tree_size else []
        pos += tree_size * 4
        (count,) = struct.unpack_from("<I", data, pos)
        pos += 4
        objects = list(struct.unpack_from(f"<{count}I", data, pos)) if count else []
        pos += count * 4

        self.bounds_lo = lo
        self.bounds_hi = hi
        self.tree = tree
        self.objects = objects
        return pos

    def intersect_ray(self, origin: Vec3, direction: Vec3, max_dist: float,
                      stop_at_first_hit: bool,
                      callback: Callable[[int, float], Tuple[bool, float]]):
        """Port of BIH::intersectRay.

        ``callback(entry, distance)`` returns ``(hit, distance)``.
        Returns ``(hit_any, max_dist)``.
        """
        tree = self.tree
        objects = self.objects

        interval_min = -1.0
        interval_max = -1.0
        inv_dir = [0.0, 0.0, 0.0]
        for i in range(3):
            d = direction[i]
            if d == 0.0:
                inv_dir[i] = math_copysign_inf(d)
            else:
                inv_dir[i] = 1.0 / d
            if fuzzy_ne32(d, 0.0):
                t1 = (self.bounds_lo[i] - origin[i]) * inv_dir[i]
                t2 = (self.bounds_hi[i] - origin[i]) * inv_dir[i]
                if t1 > t2:
                    t1, t2 = t2, t1
                if t1 > interval_min:
                    interval_min = t1
                if t2 < interval_max or interval_max < 0.0:
                    interval_max = t2
                if interval_max <= 0.0 or interval_min >= max_dist:
                    return False, max_dist

        if interval_min > interval_max:
            return False, max_dist
        interval_min = max(interval_min, 0.0)
        interval_max = min(interval_max, max_dist)

        offset_front = [0, 0, 0]
        offset_back = [0, 0, 0]
        offset_front3 = [0, 0, 0]
        offset_back3 = [0, 0, 0]
        for i in range(3):
            offset_front[i] = float_to_raw_bits(direction[i]) >> 31
            offset_back[i] = offset_front[i] ^ 1
            offset_front3[i] = offset_front[i] * 3
            offset_back3[i] = offset_back[i] * 3
            offset_front[i] += 1
            offset_back[i] += 1

        stack = [(0, 0.0, 0.0)] * MAX_STACK_SIZE
        stack_pos = 0
        node = 0
        hit_any = False

        while True:
            while True:
                tn = tree[node]
                axis = (tn & (3 << 30)) >> 30
                bvh2 = tn & (1 << 29)
                obj_offset = tn & ~(7 << 29)
                if not bvh2:
                    if axis < 3:
                        tf = (int_bits_to_float(tree[node + offset_front[axis]]) - origin[axis]) * inv_dir[axis]
                        tb = (int_bits_to_float(tree[node + offset_back[axis]]) - origin[axis]) * inv_dir[axis]
                        if tf < interval_min and tb > interval_max:
                            break
                        back = obj_offset + offset_back3[axis]
                        node = back
                        if tf < interval_min:
                            if tb >= interval_min:
                                interval_min = tb
                            continue
                        node = obj_offset + offset_front3[axis]
                        if tb > interval_max:
                            if tf <= interval_max:
                                interval_max = tf
                            continue
                        stack[stack_pos] = (back, tb if tb >= interval_min else interval_min, interval_max)
                        stack_pos += 1
                        if tf <= interval_max:
                            interval_max = tf
                        continue
                    else:
                        n = tree[node + 1]
                        while n > 0:
                            hit, max_dist = callback(objects[obj_offset], max_dist)
                            if hit:
                                hit_any = True
                            if stop_at_first_hit and hit:
                                return hit_any, max_dist
                            n -= 1
                            obj_offset += 1
                        break
                else:
                    if axis > 2:
                        return hit_any, max_dist
                    tf = (int_bits_to_float(tree[node + offset_front[axis]]) - origin[axis]) * inv_dir[axis]
                    tb = (int_bits_to_float(tree[node + offset_back[axis]]) - origin[axis]) * inv_dir[axis]
                    node = obj_offset
                    if tf >= interval_min:
                        interval_min = tf
                    if tb <= interval_max:
                        interval_max = tb
                    if interval_min > interval_max:
                        break
                    continue

            while True:
                if stack_pos == 0:
                    return hit_any, max_dist
                stack_pos -= 1
                interval_min = stack[stack_pos][1]
                if max_dist < interval_min:
                    continue
                node = stack[stack_pos][0]
                interval_max = stack[stack_pos][2]
                break

    def intersect_point(self, p: Vec3, callback: Callable[[int], None]) -> None:
        """Port of BIH::intersectPoint (callback receives tree-value index)."""
        lo, hi = self.bounds_lo, self.bounds_hi
        if not (lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1] and lo[2] <= p[2] <= hi[2]):
            return

        tree = self.tree
        objects = self.objects
        stack = [0] * MAX_STACK_SIZE
        stack_pos = 0
        node = 0

        while True:
            while True:
                tn = tree[node]
                axis = (tn & (3 << 30)) >> 30
                bvh2 = tn & (1 << 29)
                obj_offset = tn & ~(7 << 29)
                if not bvh2:
                    if axis < 3:
                        tl = int_bits_to_float(tree[node + 1])
                        tr = int_bits_to_float(tree[node + 2])
                        if tl < p[axis] and tr > p[axis]:
                            break
                        right = obj_offset + 3
                        node = right
                        if tl < p[axis]:
                            continue
                        node = obj_offset
                        if tr > p[axis]:
                            continue
                        stack[stack_pos] = right
                        stack_pos += 1
                        continue
                    else:
                        n = tree[node + 1]
                        while n > 0:
                            callback(objects[obj_offset])
                            n -= 1
                            obj_offset += 1
                        break
                else:
                    if axis > 2:
                        return
                    tl = int_bits_to_float(tree[node + 1])
                    tr = int_bits_to_float(tree[node + 2])
                    node = obj_offset
                    if tl > p[axis] or tr < p[axis]:
                        break
                    continue

            if stack_pos == 0:
                return
            stack_pos -= 1
            node = stack[stack_pos]


def math_copysign_inf(d: float) -> float:
    import math as _math
    if d == 0.0 and _math.copysign(1.0, d) < 0:
        return -_math.inf
    return _math.inf
