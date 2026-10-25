"""Motion onto an armature by rest-pose maths, written as GENERATED keys. Quaternions here are numpy x y z w."""

import bpy
import numpy as np
from bpy_extras import anim_utils
from mathutils import Matrix

from . import keys
from .rig import ARMS, CHILD, JOINTS, LEGS, PARENTS, bone_map, rest_joints

# Standing pitch (degrees above the floor) of the neck->head and ankle->ball lines: the median over 35k standing
# frames of the training clips. A rig at rest stands so; the model's rest skeleton (and the own rig) has them level.
STANDING = {12: 66.8, 7: -22.9, 8: -22.9}
FLOOR = {7: 0.04, 8: 0.04, 20: 0.02, 21: 0.02}  # the lowest an ankle or wrist goes above the floor (model metres)
ENDS = (7, 8, 10, 11, 20, 21)  # ankles, toes, wrists: kept above the floor by IK and the foot's turn, not the lift


def qmul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return np.stack([x, y, z, aw * bw - ax * bx - ay * by - az * bz], -1)


def cross(a, b):
    """np.cross of [..., 3] vectors, the same arithmetic without its per-call overhead (most of qrot's time)."""
    a0, a1, a2, b0, b1, b2 = a[..., 0], a[..., 1], a[..., 2], b[..., 0], b[..., 1], b[..., 2]
    return np.stack([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], -1)


def qinv(q):
    return q * np.array([-1, -1, -1, 1])


def qpow(q, t):
    """Unit quaternions q turned t of the way from identity (slerp)."""
    q = np.where(q[..., 3:] < 0, -q, q)
    s = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    half = np.arctan2(s, q[..., 3:]) * t
    axis = np.divide(q[..., :3], s, out=np.zeros_like(q[..., :3]), where=s > 1e-12)
    return np.concatenate([axis * np.sin(half), np.cos(half)], -1)


def qrot(q, v):
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
    w, x, y, z = Matrix(m).to_3x3().normalized().to_quaternion()
    return np.array([x, y, z, w])


def continuous(q):
    """Flip quaternions onto the hemisphere of the frame before, so keys interpolate the short way."""
    flip = np.cumprod(np.where((q[1:] * q[:-1]).sum(-1) < 0, -1.0, 1.0))
    return np.concatenate([q[:1], q[1:] * flip[:, None]])


def globals_(rot):
    g = rot.copy()
    for j, p in enumerate(PARENTS):
        if p >= 0:
            g[:, j] = qmul(g[:, p], rot[:, j])
    return g


def positions(g, root, offsets):
    p = np.zeros((*g.shape[:2], 3))
    p[:, 0] = root
    for j, par in enumerate(PARENTS):
        if par >= 0:
            p[:, j] = p[:, par] + qrot(g[:, par], offsets[j])
    return p


def resample(arr, times, quat=False):
    t = np.clip(times, 0, len(arr) - 1)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(arr) - 1)
    w = (t - i0).reshape(-1, *[1] * (arr.ndim - 1))
    a, b = arr[i0], arr[i1]
    if not quat:
        return a * (1 - w) + b * w
    b = np.where((a * b).sum(-1, keepdims=True) < 0, -b, b)
    q = a * (1 - w) + b * w
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


class Target:
    """An armature's rest pose in armature space, so the object's own location, rotation and scale apply on top."""

    def __init__(self, obj, offsets):
        self.arm, self.offsets = obj.data, offsets
        self.map = bone_map(obj)
        self.joint_of = {b: j for j, b in self.map.items() if CHILD[j] >= 0}
        self.copies = _copies(obj)
        self.order = _topological(self.arm.bones, self.copies)
        self.rest = {b.name: from_matrix(b.matrix_local) for b in self.arm.bones}
        self.axes = self._axes()
        self.q = from_matrix(self.axes)
        self.src = rest_joints(offsets)
        self.ends = {j: self._end(j) for j in self.joint_of.values()}
        self.align = {b: self._align(b, j) for b, j in self.joint_of.items()}
        self.low = self._floor()
        self.ratio = self._height() / self.src[0, 1]

    def _head(self, j):
        return np.array(self.arm.bones[self.map[j]].head_local)

    def _end(self, j):
        """The joint j's bone aims at: the next mapped joint down the chain (a rig with fewer spine bones skips joints),
        else the chain's last joint before the toe (a one-bone arm aims at the wrist, a one-bone leg at the ankle)."""
        k = CHILD[j]
        while CHILD[k] >= 0 and k not in self.map and CHILD[k] not in (10, 11):
            k = CHILD[k]
        return k

    def _dirs(self, name, j):
        """Rest direction of joint j's bone in the rig and in the model: towards the next mapped joint (control bones
        point anywhere), or along the bone at a chain's end."""
        k, src = self.ends[j], self.src @ self.axes.T
        if k in self.map:
            return self._head(k) - self._head(j), src[k] - src[j]
        bone = self.arm.bones[name]
        return np.array(bone.tail_local - bone.head_local), src[k] - src[j]

    def _align(self, name, j):
        """The turn from the rig's rest onto the model's rest. The neck and feet stand as a person stands at the rig's
        rest, so an upright cartoon head or a flat foot stays so rather than taking the model's level ones."""
        rig, model = self._dirs(name, j)
        own = all(self.map.get(i) == n for i, n in enumerate(JOINTS))
        if j not in STANDING or CHILD[j] != self.ends[j] or own:
            return arc(rig, model)
        a = np.radians(STANDING[j])
        standing = self.axes @ np.array([0, np.sin(a), np.cos(a)])
        return arc(rig, qrot(arc(standing, model), rig))

    def _axes(self):
        """Model axes (left, up, forward) in armature space, from the rig's hips and spine snapped to the nearest axis:
        Z-up rigs and Y-up imports (FBX) both work."""
        top = 12 if 12 in self.map else 15
        if not {0, 1, 2, top} <= self.map.keys():
            return np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
        left, up = _snap(self._head(1) - self._head(2)), _snap(self._head(top) - self._head(0))
        return np.column_stack([left, up, np.cross(left, up)])

    def _height(self):
        if 0 not in self.map:
            return self.src[0, 1]
        return self._head(0) @ self.axes[:, 1] - self._floor()

    def _floor(self):
        """The rig's floor height at rest: its lowest bone end."""
        return min(np.array(p) @ self.axes[:, 1] for b in self.arm.bones for p in (b.head_local, b.tail_local))

    def _world(self, rot, root):
        """World rotation per mapped joint. A bone that stands for several joints aims at its end joint: a short spine
        turns with its last joint, so the whole bend shows; a one-bone limb keeps its parent's twist."""
        g = globals_(rot)
        p = positions(g, root, self.offsets)
        out = {}
        for j, k in self.ends.items():
            base = PARENTS[k] if j in (0, 3, 6, 9, 12) else PARENTS[j]
            out[j] = g[:, j]
            if CHILD[j] != k:
                out[j] = qmul(arc(qrot(g[:, base], self.src[k] - self.src[j]), p[:, k] - p[:, j]), g[:, base])
        return {j: qmul(qmul(self.q, r), qinv(self.q)) for j, r in out.items()}

    def pose(self, rot, root, planted=None, start=None, hands=None, floor=False):
        """rot [F, 22, 4] and root [F, 3] in the model's frame -> ({bone: [F, 4] basis}, pelvis location [F, 3]).
        planted: [F, 2] left/right foot on the ground, held there by leg IK. start: (offset, yaw) to begin from.
        hands: [F, 2] left/right hand on the floor, held there at the model's height above it by arm IK. floor: no
        joint below the floor (wrists and ankles by IK, the rest by lifting the body)."""
        g = self._world(rot, root)
        absolute = {b: qmul(qmul(g[j], self.align[b]), self.rest[b]) for b, j in self.joint_of.items()}
        move = self.ratio * (root - self.src[0]) @ self.axes.T
        if start is not None:
            move = qrot(start[1], move) + start[0]
            absolute = {b: qmul(start[1], q) for b, q in absolute.items()}
        for side in range(2) if planted is not None else ():
            self._plant(absolute, move, planted[:, side], [self.map.get(j) for j in LEGS[side]])
        if hands is not None or floor:
            ground = self.low + (np.asarray(start[0]) @ self.axes[:, 1] if start is not None else 0.0)
            move = self._ground(absolute, move, ground, rot, root, hands, floor)
        posed = self._posed(absolute, len(rot))
        basis = {}
        for name in absolute:
            parent = self.arm.bones[name].parent
            local = posed[name] if parent is None else qmul(qinv(posed[parent.name]), posed[name])
            prest = self.rest[parent.name] if parent is not None else np.array([0, 0, 0, 1.0])
            basis[name] = continuous(qmul(qmul(qinv(self.rest[name]), prest), local))
        return basis, qrot(qinv(self.rest[self.map[0]]), move) if 0 in self.map else None

    def _posed(self, absolute, frames, only=None):
        posed = {}
        for bone in self.order if only is None else [b for b in self.order if b.name in only]:
            name, parent = bone.name, bone.parent
            if name in absolute:
                posed[name] = absolute[name]
            elif name in self.copies or parent is not None:
                src = self.copies.get(name) or parent.name
                posed[name] = qmul(posed[src], qmul(qinv(self.rest[src]), self.rest[name]))
            else:
                posed[name] = np.broadcast_to(self.rest[name], (frames, 4))
        return posed

    def heads(self, posed, move, only):
        """Head positions [F, 3] in armature space of the bones in only (with all they depend on); the pelvis bone's
        parent stays at rest."""
        out, pelvis = {}, self.map.get(0)
        for bone in [b for b in self.order if b.name in only]:
            h, p = np.array(bone.head_local), bone.parent
            if bone.name == pelvis or p is None:
                out[bone.name] = h + move if bone.name == pelvis else np.broadcast_to(h, move.shape)
            else:
                rel = qmul(posed[p.name], qinv(self.rest[p.name]))
                out[bone.name] = out[p.name] + qrot(rel, h - np.array(p.head_local))
        return out

    def joints(self, basis, loc, frames):
        """The way back from a pose: the bones' own rotations {bone: [F, 4]} and locations {bone: [F, 3]} (a bone left
        out is at rest) -> [F, 22, 3] joints in the model's frame. A joint the map leaves out keeps its rest offset
        from the nearest mapped joint above it."""
        rot, head, rest0, used = {}, {}, np.zeros((frames, 4)) + [0, 0, 0, 1.0], self.deps(self.map.values())
        for bone in [b for b in self.order if b.name in used]:
            name, p = bone.name, bone.parent
            at = np.array(bone.head_local) + qrot(self.rest[name], loc.get(name, np.zeros((frames, 3))))
            turn = qmul(rot[p.name], qinv(self.rest[p.name])) if p else rest0
            rot[name] = qmul(turn, qmul(self.rest[name], basis.get(name, rest0)))
            head[name] = head[p.name] + qrot(turn, at - np.array(p.head_local)) if p else at
            if name in self.copies:
                src = self.copies[name]
                rot[name] = qmul(rot[src], qmul(qinv(self.rest[src]), self.rest[name]))
        placed = self._head(0) + self.ratio * (self.src - self.src[0]) @ self.axes.T
        out = np.zeros((frames, len(PARENTS), 3))
        for j, a in enumerate(PARENTS):
            if j in self.map:
                out[:, j] = head[self.map[j]]
                continue
            while a not in self.map:
                a = PARENTS[a]
            turn = qmul(rot[self.map[a]], qinv(self.rest[self.map[a]]))
            out[:, j] = out[:, a] + qrot(turn, placed[j] - placed[a])
        return (out - self._head(0)) @ self.axes / self.ratio + self.src[0]

    def place(self, theta, shift):
        """pose's start for the model's frame turned by theta [F] about its up axis and moved by shift [F, 3]."""
        half = np.asarray(theta)[..., None] / 2
        q = np.concatenate([np.zeros_like(half), np.sin(half), np.zeros_like(half), np.cos(half)], -1)
        offset = self.ratio * (qrot(q, self.src[0]) + shift - self.src[0]) @ self.axes.T
        return offset, qmul(qmul(self.q, q), qinv(self.q))

    def deps(self, names):
        """names and every bone their pose depends on (parents, copied bones)."""
        out, todo = set(), list(names)
        while todo:
            bone = self.arm.bones[todo.pop()]
            if bone.name not in out:
                out.add(bone.name)
                todo += [d for d in (bone.parent and bone.parent.name, self.copies.get(bone.name)) if d]
        return out

    def _plant(self, absolute, move, planted, names):
        if None in names or not planted.any():
            return
        hip, knee, ankle = names
        only = self.deps(names)
        heads = self.heads(self._posed(absolute, len(move), only), move, only)
        h, k, a = heads[hip], heads[knee], heads[ankle]
        front = qrot(qmul(absolute[hip], qinv(self.rest[hip])), self.axes[:, 2])
        turn_thigh, turn_shin = two_bone(h, k, a, hold(a, planted, up=self.axes[:, 1]), front)
        absolute[hip], absolute[knee] = qmul(turn_thigh, absolute[hip]), qmul(turn_shin, absolute[knee])

    def _ground(self, absolute, move, ground, rot, root, hands, floor):
        """The floor steps of pose, in order: lift the body, hands onto the floor, limb ends and toe tips above it."""
        up = self.axes[:, 1]
        if floor:
            move = move + self._lift(absolute, move, ground)[:, None] * up
        if hands is not None:
            wrist = positions(globals_(rot), root, self.offsets)[..., 1]
            for chain, on in zip(ARMS, hands.T, strict=True):
                at = ground + self.ratio * wrist[:, chain[2]]
                self._reach(
                    absolute, move, chain, lambda a, on=on, at=at: hold(a, on, at=a + (at - a @ up)[:, None] * up)
                )
        if floor:
            for chain in LEGS + ARMS:
                lo = ground + self.ratio * FLOOR[chain[2]]
                self._reach(absolute, move, chain, lambda a, lo=lo: a + np.maximum(lo - a @ up, 0)[:, None] * up)
            for ankle in (7, 8):
                self._tiptoe(absolute, move, ankle, ground)
        return move

    def _reach(self, absolute, move, chain, aim):
        """Two-bone IK of a limb (joints chain: upper, middle, end) onto aim(end heads [F, 3]) -> targets, on the
        frames where the target moves the end. The knee or elbow keeps its bend, a little towards the front or back."""
        names = [self.map.get(j) for j in chain]
        if None in names:
            return
        upper, middle, end = names
        only = self.deps(names)
        heads = self.heads(self._posed(absolute, len(move), only), move, only)
        h, k, a = heads[upper], heads[middle], heads[end]
        t = aim(a)
        on = np.linalg.norm(t - a, axis=1) > 1e-6
        if not on.any():
            return
        fwd = qrot(qmul(absolute[upper], qinv(self.rest[upper])), self.axes[:, 2])
        turns = two_bone(h[on], k[on], a[on], t[on], (0.1 if chain in LEGS else -0.1) * fwd[on])
        for name, turn in zip((upper, middle), turns, strict=True):
            q = np.array(absolute[name])
            q[on] = qmul(turn, q[on])
            absolute[name] = q

    def _lift(self, absolute, move, ground, ease=4):
        """[F] how far to lift the body so no joint but the limb ends is below the floor, eased over a few frames."""
        names = [b for j, b in self.map.items() if j not in ENDS]
        only = self.deps(names)
        heads = self.heads(self._posed(absolute, len(move), only), move, only)
        need = np.maximum(ground - np.min([heads[b] @ self.axes[:, 1] for b in names], 0), 0)
        pad = np.pad(need, 2 * ease, mode="edge")
        peak = np.max([pad[i : i + len(need) + 2 * ease] for i in range(2 * ease + 1)], 0)
        return np.convolve(peak, np.full(2 * ease + 1, 1 / (2 * ease + 1)), "valid")

    def _tiptoe(self, absolute, move, ankle, ground):
        """A foot whose tip (toe joint, else the foot bone's tail) is below the floor turns about the ankle onto it."""
        foot, toe = self.map.get(ankle), self.map.get(ankle + 3)
        if foot is None:
            return
        only = self.deps([foot, toe] if toe else [foot])
        posed = self._posed(absolute, len(move), only)
        heads = self.heads(posed, move, only)
        bone, up = self.arm.bones[foot], self.axes[:, 1]
        tail = qrot(qmul(posed[foot], qinv(self.rest[foot])), np.array(bone.tail_local - bone.head_local))
        v = heads[toe] - heads[foot] if toe else tail
        drop = ground - heads[foot] @ up  # where the tip may go, from the ankle
        on = v @ up < drop
        if not on.any():
            return
        n = np.linalg.norm(v[on], axis=1)
        h = np.maximum(drop[on], -n)
        flat = v[on] - (v[on] @ up)[:, None] * up
        flat /= np.linalg.norm(flat, axis=1, keepdims=True) + 1e-9
        q = np.array(absolute[foot])
        q[on] = qmul(arc(v[on], flat * np.sqrt(n**2 - h**2)[:, None] + h[:, None] * up), q[on])
        absolute[foot] = q


def hold(a, planted, blend=8, up=None, at=None):
    """Ankle targets: over each planted span the mean position; the shift fades out over a few frames (smoothstep) and
    carries across short gaps, so a target never jumps. up: only the floor position is held, the ankle keeps its own
    height (the heel and toe roll), so the leg is not pulled straight. at: the positions to hold in place of a's own."""
    f, idx = np.arange(len(a)), np.flatnonzero(planted)
    if not len(idx):
        return a
    held = a.copy()
    edges = np.flatnonzero(np.diff(np.r_[0, planted.astype(int), 0]))
    for s, e in zip(edges[::2], edges[1::2], strict=True):
        held[s:e] = (a if at is None else at)[s:e].mean(0)
    shift = np.stack([np.interp(f, idx, (held - a)[idx, i]) for i in range(3)], 1)
    if up is not None:
        shift -= (shift @ up)[:, None] * up
    w = np.clip(1 - np.abs(f[:, None] - idx[None]).min(1) / (blend + 1), 0, 1)
    return a + (w * w * (3 - 2 * w))[:, None] * shift


def two_bone(h, k, a, t, front):
    """Analytic two-bone IK per frame: hip h, knee k, ankle a, ankle target t -> (thigh turn, shin turn) as world
    quaternions. The knee bends forward, turned with the hip, and towards where it was; a soft limit at 99% of the reach
    keeps the leg from snapping straight."""
    l1, l2 = np.linalg.norm(k - h, axis=1), np.linalg.norm(a - k, axis=1)
    d = t - h
    reach, dist = l1 + l2, np.linalg.norm(d, axis=1)
    soft = 0.99 * reach
    over = np.maximum(dist - soft, 0)
    dist = np.maximum(np.minimum(dist, soft) + (reach - soft) * (1 - np.exp(-over / (reach - soft))), np.abs(l1 - l2))
    u = d / np.linalg.norm(d, axis=1, keepdims=True)
    pole = (k - h) + l1[:, None] * front  # knees bend forward: a nearly straight leg cannot flip back
    pole -= (pole * u).sum(1, keepdims=True) * u
    v = pole / np.linalg.norm(pole, axis=1, keepdims=True)
    c = np.clip((l1**2 + dist**2 - l2**2) / (2 * l1 * dist), -1, 1)[:, None]
    knee = h + l1[:, None] * (c * u + np.sqrt(1 - c**2) * v)
    turn_thigh = arc(k - h, knee - h)
    return turn_thigh, qmul(arc(qrot(turn_thigh, a - k), h + u * dist[:, None] - knee), turn_thigh)


def hand_contacts(joints, fps, height=0.09, speed=0.4):
    """[F, 2] left/right wrist on the floor in a motion (joints [F, 22, 3], metres, Y up): low and nearly still."""
    w = joints[:, [20, 21]]
    v = np.linalg.norm(np.diff(w[..., [0, 2]], axis=0, prepend=w[:1, :, [0, 2]]), axis=-1) * fps
    return (w[..., 1] < height) & (v < speed)


def _snap(v):
    axis = np.zeros(3)
    k = int(np.argmax(np.abs(v)))
    axis[k] = np.sign(v[k])
    return axis


def _copies(obj):
    """Bone -> the bone whose rotation it copies (Copy Rotation / Copy Transforms on the same armature), so control
    rigs that isolate a limb or the neck from their parent are followed as Blender evaluates them."""
    out = {}
    for pb in obj.pose.bones:
        for c in pb.constraints:
            if c.type in {"COPY_ROTATION", "COPY_TRANSFORMS"} and c.target == obj and c.subtarget and c.influence > 0.5:
                out[pb.name] = c.subtarget
    return out


def _topological(bones, copies):
    out, seen = [], set()

    def visit(b):
        if b.name not in seen:
            seen.add(b.name)
            for dep in (b.parent, bones.get(copies.get(b.name, ""))):
                if dep is not None:
                    visit(dep)
            out.append(b)

    for b in bones:
        visit(b)
    return out


def start_pose(obj, target, frame):
    """(offset, yaw) in armature space: the root bone's own keyed pose (not GENERATED) at or before frame, so a take
    continues from where the user left the character. None without such keys."""
    name, ad = target.map.get(0), obj.animation_data
    if name is None or ad is None or ad.action is None or ad.action_slot is None:
        return None
    cb, values = channelbag(obj), {}
    for path, n in (("location", 3), ("rotation_quaternion", 4)):
        for i in range(n):
            fc = cb.fcurves.find(f'pose.bones["{name}"].{path}', index=i)
            co, kind = keys.get(fc.keyframe_points) if fc else (np.zeros((0, 2)), np.zeros(0))
            own = co[(kind != keys.GENERATED) & (co[:, 0] <= frame)]
            if len(own):
                values[path, i] = own[np.argmax(own[:, 0]), 1]
    if not values:
        return None
    loc = np.array([values.get(("location", i), 0.0) for i in range(3)])
    w, x, y, z = (values.get(("rotation_quaternion", i), 1.0 if i == 0 else 0.0) for i in range(4))
    rest = target.rest[name]
    turn = qmul(qmul(rest, np.array([x, y, z, w]) / np.linalg.norm([x, y, z, w])), qinv(rest))
    fwd, up = target.axes[:, 2], target.axes[:, 1]
    flat = qrot(turn, fwd)
    flat -= (flat @ up) * up
    return qrot(rest, loc) - (qrot(rest, loc) @ up) * up, arc(fwd, flat)


def channelbag(obj, action=None):
    """A linked or overridden action (library overrides start with one) is replaced by a local copy."""
    if action is not None:
        slot = action.slots.get("OB" + obj.name) or action.slots.new("OBJECT", obj.name)
        return anim_utils.action_ensure_channelbag_for_slot(action, slot)
    ad = obj.animation_data or obj.animation_data_create()
    if ad.action is None or ad.action.library is not None or ad.action.override_library is not None:
        ad.action = ad.action.copy() if ad.action else bpy.data.actions.new(obj.name + "Action")
    if ad.action_slot is None:
        ad.action_slot = ad.action.slots.new("OBJECT", obj.name)
    return anim_utils.action_ensure_channelbag_for_slot(ad.action, ad.action_slot)


def write(obj, frames, basis, location=None, root_bone=None, fit=None, action=None, kind=None):
    """The take onto obj's action. fit: (degrees, cm, marked frames, shared, {bone: frames to hold 8x tighter}) to
    write it unbaked, else every frame. Planted feet need the tight frames: small leg and hip errors make them slide."""
    cb, groups = channelbag(obj, action), []
    for name, q in basis.items():
        if obj.pose.bones[name].rotation_mode != "QUATERNION":
            obj.pose.bones[name].rotation_mode = "QUATERNION"
        groups.append((f'pose.bones["{name}"].rotation_quaternion', name, q[:, [3, 0, 1, 2]], keys.angle))
    if location is not None:
        groups.append((f'pose.bones["{root_bone}"].location', root_bone, location, keys.distance))
    if fit is None:
        picks, slopes = [[np.arange(len(frames))] * g[2].shape[1] for g in groups], [None] * len(groups)
    else:
        degrees, cm, marks, shared, tight = fit
        limit = {keys.angle: np.radians(degrees), keys.distance: cm / 100 / max(obj.matrix_world.to_scale())}
        limits = [limit[fn] / np.where(tight.get(name, False), 8, 1) for _, name, _, fn in groups]
        picks, slopes = keys.unbake(
            frames, [(g[2], g[3], lim) for g, lim in zip(groups, limits, strict=True)], marks, shared
        )
    cut = np.flatnonzero(np.diff(frames) > 1)
    spans = list(
        zip(frames[np.r_[0, cut + 1]], frames[np.r_[cut, len(frames) - 1]], strict=True)
    )  # own keys in gaps stay
    for (path, name, values, _), ks, m in zip(groups, picks, slopes, strict=True):
        for i, k in enumerate(ks):
            fc = cb.fcurves.ensure(path, index=i, group_name=name)
            keys.write(fc, frames[k], values[k, i], None if m is None else m[k, i], kind or keys.GENERATED, spans)


def to_rest(frames, basis, loc, own, n, hold=False, up=None):
    """Frames after each stretch of the take where nothing follows (no block, no own key in own): the bones ease to
    rest with a smoothstep over n frames and hold it (a frame more: a flat end). Into a later block they ease back
    over the same time, or over half the gap if it is shorter. hold: the stretch's last pose instead of rest, only in
    gaps. up: the root bone's local up axis; its location eases to rest only along it, so the rig stays where it
    walked to. -> (frames, {bone: [F, 4]}, location [F, 3] or None, frames to key: each stretch's end and where a hold
    starts and ends)."""
    ends = np.r_[np.flatnonzero(np.diff(frames) > 1), len(frames) - 1]
    out, marks = [], []
    for i in ends:
        e = frames[i]
        s = frames[i + 1] if i + 1 < len(frames) else None
        if ((own > e) & (own < (np.inf if s is None else s))).any() or (hold and s is None):
            continue
        f = np.arange(e + 1, e + n + 2 if s is None else s)
        d = n if s is None else min(n, (s - e) / 2)
        near = np.minimum(f - e, np.inf if s is None else s - f)
        t = np.clip(near / d, 0, 1)
        w = t * t * (3 - 2 * t)
        side = np.where(f - e <= near, i, i + 1)  # the pose each frame eases from: the stretch's end or what follows
        out.append((f, w, side, np.full(len(f), i)))
        full = w == 1
        inner = full & np.r_[False, full[:-1]] & np.r_[full[1:], False]
        flat = inner if inner.any() else full  # keys a frame into the hold: their slope is zero there
        marks += [e, *f[flat & ~(np.r_[False, flat[:-1]] & np.r_[flat[1:], False])]]
    if not out:
        return frames, basis, loc, []
    f, w, side, end = (np.concatenate(x) for x in zip(*out, strict=True))
    order = np.argsort(np.r_[frames, f], kind="stable")

    def eased(v, rest):
        rest = v[end] if hold else rest
        r = np.where((v[side] * rest).sum(1, keepdims=True) < 0, -rest, rest) if v.shape[1] == 4 else rest
        x = v[side] * (1 - w[:, None]) + r * w[:, None]
        x = x / np.linalg.norm(x, axis=1, keepdims=True) if v.shape[1] == 4 else x
        return np.concatenate([v, x])[order]

    unit = np.array([0, 0, 0, 1.0])
    rested = {name: eased(q, unit) for name, q in basis.items()}
    if loc is not None:
        loc = eased(loc, np.zeros(3) if up is None else loc[end] - (loc[end] @ up)[:, None] * up)
    return np.r_[frames, f][order], rested, loc, marks
