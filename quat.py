"""Quaternion maths on numpy arrays [..., 4] in x y z w order."""

import numpy as np
from mathutils import Matrix


def qmul(a, b):
    """Quaternion product a b."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return np.stack([x, y, z, aw * bw - ax * bx - ay * by - az * bz], -1)


def cross(a, b):
    """np.cross of [..., 3] vectors without its per-call overhead."""
    a0, a1, a2, b0, b1, b2 = a[..., 0], a[..., 1], a[..., 2], b[..., 0], b[..., 1], b[..., 2]
    return np.stack([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], -1)


def qinv(q):
    """Inverse of a unit quaternion."""
    return q * np.array([-1, -1, -1, 1])


def qpow(q, t):
    """Unit quaternions q turned t of the way from identity (slerp)."""
    q = np.where(q[..., 3:] < 0, -q, q)
    s = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    half = np.arctan2(s, q[..., 3:]) * t
    axis = np.divide(q[..., :3], s, out=np.zeros_like(q[..., :3]), where=s > 1e-12)
    return np.concatenate([axis * np.sin(half), np.cos(half)], -1)


def qrot(q, v):
    """Vectors v rotated by q."""
    u, w = q[..., :3], q[..., 3:]
    return v + 2 * cross(u, cross(u, v) + w * v)


def arc(a, b):
    """The shortest rotation turning direction a onto direction b (both [..., 3])."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    q = np.concatenate([cross(a, b), 1 + (a * b).sum(-1, keepdims=True)], -1)
    side = np.where(np.abs(a[..., :1]) < 0.9, [1.0, 0, 0], [0, 1.0, 0])
    q = np.where(q[..., 3:] < 1e-8, np.concatenate([cross(a, side), np.zeros_like(q[..., :1])], -1), q)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def from_matrix(m):
    """A matrix's rotation as x y z w."""
    w, x, y, z = Matrix(m).to_3x3().normalized().to_quaternion()
    return np.array([x, y, z, w])


def continuous(q):
    """Flip quaternions onto the hemisphere of the frame before, so keys interpolate the short way."""
    flip = np.cumprod(np.where((q[1:] * q[:-1]).sum(-1) < 0, -1.0, 1.0))
    return np.concatenate([q[:1], q[1:] * flip[:, None]])
