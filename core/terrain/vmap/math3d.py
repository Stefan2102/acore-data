"""Minimal ports of the G3D math used by AzerothCore's vmap collision.

Only the exact operations needed by the VMAP raycast are implemented, with
the same semantics as the vendored G3D (deps/g3dlite).  Vectors are plain
3-tuples; 3x3 matrices are row-major 9-tuples.
"""

from __future__ import annotations

import math
import struct
from typing import Callable, Tuple

Vec3 = Tuple[float, float, float]
Mat3 = Tuple[float, ...]

FUZZY_EPSILON32 = 1e-5


def _eps32(a: float) -> float:
    """G3D eps(float a, float b): fuzzyEpsilon32 * (|a| + 1)."""
    aa = abs(a) + 1.0
    if math.isinf(aa):
        return FUZZY_EPSILON32
    return FUZZY_EPSILON32 * aa


def fuzzy_eq32(a: float, b: float) -> bool:
    return a == b or abs(a - b) <= _eps32(a)


def fuzzy_ne32(a: float, b: float) -> bool:
    return not fuzzy_eq32(a, b)


def fuzzy_ge32(a: float, b: float) -> bool:
    """G3D fuzzyGe(float): a > b - eps(a, b)."""
    return a > b - _eps32(a)


def fuzzy_lt32(a: float, b: float) -> bool:
    """G3D fuzzyLt(float): a < b + eps(a, b)."""
    return a < b + _eps32(a)


def float_to_raw_bits(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def int_bits_to_float(i: int) -> float:
    return struct.unpack("<f", struct.pack("<I", i & 0xFFFFFFFF))[0]


def v_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_mul(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def v_mad(a: Vec3, b: Vec3, s: float) -> Vec3:
    return (a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s)


def v_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def v_length(a: Vec3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def v_normalize(a: Vec3) -> Vec3:
    n = v_length(a)
    if n == 0.0:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def mat3_mul(a: Mat3, b: Mat3) -> Mat3:
    out = []
    for r in range(3):
        for c in range(3):
            out.append(
                a[r * 3 + 0] * b[0 * 3 + c]
                + a[r * 3 + 1] * b[1 * 3 + c]
                + a[r * 3 + 2] * b[2 * 3 + c]
            )
    return tuple(out)


def mat3_vec(m: Mat3, v: Vec3) -> Vec3:
    return (
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    )


def mat3_transpose(m: Mat3) -> Mat3:
    return (m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8])


def mat3_inverse(m: Mat3, tolerance: float = 1e-6) -> Mat3:
    """Port of G3D Matrix3::inverse (cofactors)."""
    inv = [
        m[4] * m[8] - m[5] * m[7],
        m[2] * m[7] - m[1] * m[8],
        m[1] * m[5] - m[2] * m[4],
        m[5] * m[6] - m[3] * m[8],
        m[0] * m[8] - m[2] * m[6],
        m[2] * m[3] - m[0] * m[5],
        m[3] * m[7] - m[4] * m[6],
        m[1] * m[6] - m[0] * m[7],
        m[0] * m[4] - m[1] * m[3],
    ]
    det = m[0] * inv[0] + m[1] * inv[3] + m[2] * inv[6]
    if abs(det) <= tolerance:
        return mat3_transpose(m)
    inv_det = 1.0 / det
    return tuple(v * inv_det for v in inv)


def mat3_from_euler_zyx(y_angle: float, p_angle: float, r_angle: float) -> Mat3:
    """G3D Matrix3::fromEulerAnglesZYX (row-major Rz * Ry * Rx)."""
    cy, sy = math.cos(y_angle), math.sin(y_angle)
    kz = (cy, -sy, 0.0, sy, cy, 0.0, 0.0, 0.0, 1.0)

    cp, sp = math.cos(p_angle), math.sin(p_angle)
    ky = (cp, 0.0, sp, 0.0, 1.0, 0.0, -sp, 0.0, cp)

    cr, sr = math.cos(r_angle), math.sin(r_angle)
    kx = (1.0, 0.0, 0.0, 0.0, cr, -sr, 0.0, sr, cr)

    return mat3_mul(kz, mat3_mul(ky, kx))


def aabb_contains(lo: Vec3, hi: Vec3, p: Vec3) -> bool:
    return (
        p[0] >= lo[0] and p[0] <= hi[0]
        and p[1] >= lo[1] and p[1] <= hi[1]
        and p[2] >= lo[2] and p[2] <= hi[2]
    )


def ray_intersection_time_aabb(origin: Vec3, direction: Vec3,
                               lo: Vec3, hi: Vec3) -> float:
    """Port of G3D Ray::intersectionTime(AABox).

    Returns the hit distance along the ray (``inf`` for a miss) and 0.0 when
    the origin is inside the box and no earlier hit exists.
    """
    inside = True
    max_t = [-1.0, -1.0, -1.0]
    location = [0.0, 0.0, 0.0]

    for i in range(3):
        if origin[i] < lo[i]:
            location[i] = lo[i]
            inside = False
            if float_to_raw_bits(direction[i]) != 0:
                max_t[i] = (lo[i] - origin[i]) / direction[i]
        elif origin[i] > hi[i]:
            location[i] = hi[i]
            inside = False
            if float_to_raw_bits(direction[i]) != 0:
                max_t[i] = (hi[i] - origin[i]) / direction[i]

    if inside:
        # collisionLocationForMovingPointFixedAABox returns false; the
        # Ray::intersectionTime wrapper turns that into 0.0 for inside rays.
        return 0.0

    which = 0
    if max_t[1] > max_t[which]:
        which = 1
    if max_t[2] > max_t[which]:
        which = 2

    if float_to_raw_bits(max_t[which]) & 0x80000000:
        return math.inf

    for i in range(3):
        if i == which:
            continue
        location[i] = origin[i] + max_t[which] * direction[i]
        if location[i] < lo[i] or location[i] > hi[i]:
            return math.inf

    hx = location[0] - origin[0]
    hy = location[1] - origin[1]
    hz = location[2] - origin[2]
    return math.sqrt(hx * hx + hy * hy + hz * hz)


def intersect_triangle(points, tri, ray_origin: Vec3, ray_dir: Vec3,
                       distance: float):
    """Port of VMAP::IntersectTriangle (WorldModel.cpp).

    ``tri`` indexes into ``points``; ``distance`` is the current closest hit.
    Returns ``(hit, distance)``.
    """
    p0 = points[tri[0]]
    p1 = points[tri[1]]
    p2 = points[tri[2]]

    e1 = v_sub(p1, p0)
    e2 = v_sub(p2, p0)
    p = v_cross(ray_dir, e2)
    a = v_dot(e1, p)

    if abs(a) < 1e-5:
        return False, distance

    f = 1.0 / a
    s = v_sub(ray_origin, p0)
    u = f * v_dot(s, p)
    if u < 0.0 or u > 1.0:
        return False, distance

    q = v_cross(s, e1)
    v = f * v_dot(ray_dir, q)
    if v < 0.0 or (u + v) > 1.0:
        return False, distance

    t = f * v_dot(e2, q)
    if 0.0 < t < distance:
        return True, t
    return False, distance


def ray_intersection_time_callback(origin: Vec3, direction: Vec3,
                                   lo: Vec3, hi: Vec3,
                                   intersect: Callable[[Vec3, Vec3, float], Tuple[bool, float]],
                                   distance: float):
    """Ray vs AABB where a hit delegates to ``intersect(origin, direction, distance)``.

    Mirrors ModelInstance::intersectRay's bound test followed by the model
    intersection.
    """
    time = ray_intersection_time_aabb(origin, direction, lo, hi)
    if math.isinf(time):
        return False, distance
    return intersect(origin, direction, distance)
