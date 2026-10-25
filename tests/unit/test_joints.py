"""The way back from a rig's pose to the model's joints (retarget.Target.joints and place) on a synthetic own rig."""

import numpy as np
import pytest
from sprongmotion import retarget as r
from sprongmotion import rig

OFFSETS = np.array([
    [0, 0, 0], [0.1, -0.08, 0], [-0.1, -0.08, 0], [0, 0.1, -0.02], [0, -0.4, 0.01], [0, -0.4, 0.01], [0, 0.13, 0],
    [0, -0.4, -0.03], [0, -0.4, -0.03], [0, 0.05, 0.02], [0, -0.05, 0.12], [0, -0.05, 0.12], [0, 0.2, -0.02],
    [0.08, 0.1, 0], [-0.08, 0.1, 0], [0, 0.1, 0.03], [0.12, 0, 0], [-0.12, 0, 0], [0.25, 0, 0], [-0.25, 0, 0],
    [0.25, 0, 0], [-0.25, 0, 0],
])  # fmt: skip


class Bone:
    def __init__(self, name, head, tail, parent):
        self.name, self.head_local, self.tail_local, self.parent = name, head, tail, parent


class Bones(dict):
    def __iter__(self):
        return iter(self.values())


def target(scale=1.0, drop=()):
    """The own rig (Z up, the arms in an A pose) at scale, with the joints in drop left out of the map."""
    p = rig.own_rig_joints(OFFSETS) * scale
    bones = Bones()
    for j, name in enumerate(rig.JOINTS):
        tail = p[rig.CHILD[j]] if rig.CHILD[j] >= 0 else p[j] + [0, 0, 0.05 * scale]
        bones[name] = Bone(name, p[j], tail, bones.get(rig.JOINTS[rig.PARENTS[j]]) if j else None)
    t = object.__new__(r.Target)
    t.arm, t.offsets, t.copies = type("Arm", (), {"bones": bones})(), OFFSETS, {}
    t.map = {j: n for j, n in enumerate(rig.JOINTS) if n not in drop}
    t.joint_of = {b: j for j, b in t.map.items() if rig.CHILD[j] >= 0}
    t.order = r._topological_order(bones, {})
    t.rest = {b.name: r.arc(np.array([0, 1.0, 0]), b.tail_local - b.head_local) for b in bones}
    t.axes = t._model_axes()
    t.q = _quat(t.axes)
    t.src = rig.rest_joints(OFFSETS)
    t.ends = {j: t._aim_joint(j) for j in t.joint_of.values()}
    t.align = {b: r.arc(*t._rest_directions(b, j)) for b, j in t.joint_of.items()}
    t.ratio = t._height() / t.src[0, 1]
    return t


def _quat(m):
    """A rotation matrix -> x y z w (trace method; the axes here are a quarter turn about x)."""
    w = np.sqrt(max(0.0, 1 + np.trace(m))) / 2
    return np.array([(m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w), w])


def motion(frames=12, seed=0):
    rng = np.random.default_rng(seed)
    rot = np.concatenate([rng.normal(0, 0.25, (frames, 22, 3)), np.ones((frames, 22, 1))], -1)
    rot /= np.linalg.norm(rot, axis=-1, keepdims=True)
    root = np.cumsum(rng.normal(0, 0.05, (frames, 3)), 0) + rig.rest_joints(OFFSETS)[0]
    return rot, root


def back(t, rot, root, start=None):
    basis, loc = t.pose(rot, root, start=start)
    return t.joints(basis, {t.map[0]: loc}, len(rot))


@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_round_trip(scale):
    t, (rot, root) = target(scale), motion()
    want = r.positions(r.globals_(rot), root, OFFSETS)
    assert abs(t.ratio - scale) < 1e-9
    assert np.abs(back(t, rot, root) - want).max() < 1e-9


def test_place_turns_and_moves_the_motion():
    t, (rot, root) = target(1.7), motion()
    theta, shift = np.full(len(rot), 0.8), np.tile([0.3, 0, -1.2], (len(rot), 1))
    c, s = np.cos(0.8), np.sin(0.8)
    turn = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    want = r.positions(r.globals_(rot), root, OFFSETS) @ turn.T + shift[:, None]
    assert np.abs(back(t, rot, root, t.place(theta, shift)) - want).max() < 1e-9


def test_unmapped_joints_sit_at_their_rest_offsets():
    drop = ("spine2", "l_collar", "r_collar", "l_foot", "r_foot")
    t, (rot, root) = target(drop=drop), motion()
    rest = np.tile([0, 0, 0, 1.0], (1, 22, 1))
    assert np.abs(back(t, rest, root[:1]) - r.positions(rest, root[:1], OFFSETS)).max() < 1e-9
    got = back(t, rot, root)
    for j in (rig.JOINTS.index(n) for n in drop):
        p = rig.PARENTS[j] if rig.JOINTS[rig.PARENTS[j]] not in drop else rig.PARENTS[rig.PARENTS[j]]
        rest = np.linalg.norm(t.src[j] - t.src[p])
        assert np.allclose(np.linalg.norm(got[:, j] - got[:, p], axis=-1), rest), rig.JOINTS[j]
