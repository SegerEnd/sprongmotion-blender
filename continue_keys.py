"""Continue from Keys and Fill Gap: the rig's own keys next to the timeline join the generate call as context
blocks, so the take leads out of and into them. The keys themselves are never written.
"""

import hashlib
from collections import namedtuple

import bpy
import numpy as np
from bpy_extras import anim_utils

from . import blocks, generate, keyframes, model, quat, retarget, rig, rig_data

NEAR = 2  # frames: a block starts where the keys end (or ends where they start again) within this
SECONDS = 2.0  # of keys read on each side at most
FIT = 10  # model frames next to a seam that place the take's root
FADE = 0.4  # seconds over which the take eases out of the keys' pose at a seam
NEEDS = (0, 1, 2, 4, 5, 7, 8)  # pelvis, hips, knees, ankles
# the far seam's root may be off by WARN m per second of gap plus STRIDE of the travel (turned by WARN_TURN deg/s)
# before the block warns. On 1 s gaps cut from takes, natural ones end 0-0.46 m and 0-33 deg off; moving the keys after
# by 0.25 / 0.5 / 1 m raises foot slide from 15 to 19 / 31 / 61 cm/s.
WARN, STRIDE, WARN_TURN = 0.25, 0.1, 30.0
DEBOUNCE = 0.3

Spec = namedtuple("Spec", "seam times key")  # which keys a side reads: its seam frame, the scene frames, their hash
# A context block for timeline_motion.run_blocks, and what joins it to the keys: the seam, the keys' joints at its
# decoded frames [frames, 22, 3] and their own pose at the seam ({bone: quat}, {bone: loc})
Side = namedtuple("Side", "uid text seed codes frames seam joints pose key")


class _State:
    def __init__(self):
        self.cache = {}  # key -> Side
        self.due = {}  # rig session_uid -> when to look at its keys again
        self.seen = {}  # rig session_uid -> the block keys last seen


_state = _State()


def why_not(obj, m=None):
    """Why the rig's keys cannot join its blocks, None if they can. Reads only (draws call it)."""
    s = blocks.settings(obj)
    if m is not None and not m.encoder:
        return "This model file has no motion encoder"
    if s and s.loop:
        return "Not with Loop: a cycle has no keys before or after it"
    if s and s.path:
        return "Not with a path yet"
    if len(obj.sprongmotion.bones) and not set(NEEDS) <= rig.bone_map(obj).keys():
        return "Needs the hips and legs mapped (the Rig panel)"
    return None


def channelbag(obj):
    """The rig's action's channels for reading, None without any (creates nothing)."""
    ad = obj.animation_data
    if ad is None or ad.action is None or ad.action_slot is None:
        return None
    return anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)


def own_frames(obj, names=None):
    """Sorted frames with the rig's own keys (any but the add-on's) on its mapped bones that are not frozen."""
    cb = channelbag(obj)
    if cb is None:
        return np.zeros(0)
    names = (set(rig.bone_map(obj).values()) if names is None else names) - rig_data.frozen(obj)
    found = [np.zeros(0)]
    for fc in cb.fcurves:
        if keyframes.bone_name(fc) in names:
            co, kind = keyframes.get(fc.keyframe_points)
            found.append(co[kind != keyframes.GENERATED, 0])
    return np.unique(np.concatenate(found))


def gap(obj, frame):
    """(last own key frame at or before frame, first after it) when the rig has own keys on both sides, else None."""
    f = own_frames(obj)
    lo, hi = f[f <= frame], f[f > frame]
    return (int(round(lo[-1])), int(round(hi[0]))) if len(lo) and len(hi) else None


def selected_gap(obj, frame):
    """(start, end) between the rig's selected keys around frame, else the widest such gap. None without two."""
    cb = channelbag(obj)
    picked = [np.zeros(0)]
    for fc in cb.fcurves if cb else ():
        kp = fc.keyframe_points
        on = np.zeros(len(kp), bool)
        kp.foreach_get("select_control_point", on)
        picked.append(keyframes.get(kp)[0][on, 0])
    f = np.unique(np.round(np.concatenate(picked)))
    if len(f) < 2:
        return None
    pairs = list(zip(f[:-1], f[1:], strict=True))
    around = [p for p in pairs if p[0] <= frame <= p[1]]
    a, b = around[0] if around else max(pairs, key=lambda p: p[1] - p[0])
    return int(a), int(b)


def clear(obj, lo, hi):
    """Remove the rig's own keys strictly between lo and hi on its mapped bones that are not frozen."""
    cb, names = channelbag(obj), set(rig.bone_map(obj).values()) - rig_data.frozen(obj)
    for fc in cb.fcurves if cb else ():
        if keyframes.bone_name(fc) in names:
            co, _ = keyframes.get(fc.keyframe_points)
            keyframes.remove(fc.keyframe_points, (co[:, 0] > lo + 0.5) & (co[:, 0] < hi - 0.5))
            fc.update()


def plan(obj, m):
    """(before, after) Specs: the keys that join the timeline's first and last block, None where none do."""
    s = blocks.settings(obj)
    line = blocks.timeline_blocks(s) if s else []
    if not line or m is None or why_not(obj, m) or not len(obj.sprongmotion.bones):
        return None, None
    frames, fps = own_frames(obj), blocks.fps()
    step, least, most = fps / m.fps, blocks.MIN_SECONDS * fps, int(SECONDS * m.fps / 2)
    out = [None, None]
    if any(((frames > b.start + NEAR) & (frames < b.end - NEAR)).any() for b in line):
        return None, None  # keys under a block: the take overwrites them, they end nowhere near
    near = frames[frames <= line[0].start + NEAR]
    if line[0].continues and len(near) and near[-1] >= line[0].start - NEAR and near[-1] - near[0] >= least:
        tokens = min(most, int((near[-1] - near[0]) / step // 2))
        times = np.minimum(line[0].start - step * np.arange(2 * tokens, -1, -1), near[-1])
        out[0] = _read_spec(obj, "before", near[-1], times)
    near = frames[frames >= line[-1].end - NEAR]
    if line[-1].continues and len(near) and near[0] <= line[-1].end + NEAR and near[-1] - near[0] >= least:
        tokens = min(most, int((near[-1] - near[0]) / step // 2))
        out[1] = _read_spec(obj, "after", near[0], np.maximum(line[-1].end + step * np.arange(2 * tokens + 1), near[0]))
    return tuple(out)


def _read_spec(obj, side, seam, times):
    """A Spec whose hash covers what the read depends on: the map, the frozen bones and the nearby own keys."""
    parts = [
        side,
        float(seam),
        times.round(4).tobytes(),
        sorted(rig.bone_map(obj).items()),
        sorted(rig_data.frozen(obj)),
    ]
    cb = channelbag(obj)
    for fc in cb.fcurves if cb else ():
        if keyframes.bone_name(fc) is not None:
            co, kind = keyframes.get(fc.keyframe_points)
            co = co[kind != keyframes.GENERATED]
            f = co[:, 0]
            lo, hi = (times[0], seam) if side == "before" else (seam, times[-1])
            first, last = np.searchsorted(f, lo, "right") - 1, np.searchsorted(f, hi, "left") + 1
            parts += [fc.data_path, fc.array_index, co[max(first, 0) : last].tobytes()]
    return Spec(float(seam), times, hashlib.sha1(repr(parts).encode()).hexdigest()[:16])


def sides(obj, m):
    """(before, after) Sides for the timeline as it is now, None where no keys join it. Cached per state of the keys."""
    specs = plan(obj, m)
    if not any(specs):
        return None, None
    target = retarget.Target(obj, m.offsets)
    out = []
    for uid, spec in zip((-1, -2), specs, strict=True):
        if spec is not None and spec.key not in _state.cache:
            if len(_state.cache) > 32:
                _state.cache.clear()
            _state.cache[spec.key] = _read_side(obj, target, m, uid, spec)
        out.append(_state.cache[spec.key] if spec else None)
    return tuple(out)


def block_keys(s, found):
    """{block uid: the keys its take joins, "before|after" hashes} for the first and last block of the timeline."""
    line = blocks.timeline_blocks(s)
    if not line:
        return {}
    out = {b.uid: "|" for b in line}
    before, after = (f.key if f else "" for f in found)
    out[line[0].uid] = before + "|"
    out[line[-1].uid] = out[line[-1].uid][:-1] + "|" + after
    return out


def _read_side(obj, target, m, uid, spec):
    """The keys at spec's frames as a Side: joints, codes and the own pose at the seam and the frame past it."""
    before = uid == -1
    times = np.r_[spec.times, spec.seam + (-1 if before else 1), spec.seam]
    basis, loc, own = _sample_own_keys(obj, target, times, spec.seam, before)
    joints = target.joints(basis, loc, len(times))[:-2]
    codes = m.encode(joints)
    pose = {n: basis[n][-2:] for n in own if n in basis}, {n: loc[n][-2:] for n in own if n in loc}
    return Side(uid, "", 0, codes, 2 * len(codes), spec.seam, joints[: 2 * len(codes)], pose, spec.key)


def _sample_own_keys(obj, target, times, seam, before):
    """The bones' (basis, location, bones with own keys) at times from the rig's own keys alone, held past the seam."""
    cb, basis, loc, own = channelbag(obj), {}, {}, set()
    for name in target.deps(target.map.values()):
        mode = obj.pose.bones[name].rotation_mode
        euler = mode not in {"QUATERNION", "AXIS_ANGLE"}
        rot, had_rot = _own_key_values(
            cb, name, "rotation_euler" if euler else "rotation_quaternion", times, seam, before
        )
        basis[name] = (
            _euler_to_quat(rot, mode) if euler else rot[:, [1, 2, 3, 0]] / np.linalg.norm(rot, axis=1)[:, None]
        )
        loc[name], had_loc = _own_key_values(cb, name, "location", times, seam, before)
        if had_rot or had_loc:
            own.add(name)
    return basis, loc, own


def _own_key_values(cb, name, prop, times, seam, before):
    """One bone property's own-key values at times, held past the seam; and whether it had any keys there."""
    n = 4 if prop == "rotation_quaternion" else 3
    out, had = np.zeros((len(times), n)), False
    if prop == "rotation_quaternion":
        out[:, 0] = 1
    for i in range(n):
        fc = cb.fcurves.find(f'pose.bones["{name}"].{prop}', index=i) if cb else None
        co, kind = keyframes.get(fc.keyframe_points) if fc else (np.zeros((0, 2)), np.zeros(0))
        f = co[kind != keyframes.GENERATED, 0]
        f = f[f <= seam + 0.5] if before else f[f >= seam - 0.5]
        if len(f):
            held = np.minimum(times, f.max()) if before else np.maximum(times, f.min())
            out[:, i], had = [fc.evaluate(t) for t in held], True
    return out, had


def _euler_to_quat(e, order):
    """Blender's Euler angles [F, 3] (order "XYZ": X first) -> x y z w."""
    q = np.zeros((len(e), 4)) + [0, 0, 0, 1.0]
    for axis in order:
        k, turn = "XYZ".index(axis), np.zeros((len(e), 4))
        turn[:, k], turn[:, 3] = np.sin(e[:, k] / 2), np.cos(e[:, k] / 2)
        q = quat.qmul(turn, q)
    return q


def fit(keys_, take):
    """(theta about the up axis, shift [3]) that put take's joints [F, 22, 3] onto keys_ in the ground plane."""
    a, b = take[..., [0, 2]].reshape(-1, 2), keys_[..., [0, 2]].reshape(-1, 2)
    ma, mb = a.mean(0), b.mean(0)
    a, b = a - ma, b - mb
    theta = np.arctan2((a[:, 1] * b[:, 0] - a[:, 0] * b[:, 1]).sum(), (a * b).sum())
    c, s = np.cos(theta), np.sin(theta)
    return theta, np.array([mb[0] - (c * ma[0] + s * ma[1]), 0, mb[1] - (-s * ma[0] + c * ma[1])])


def turned(theta, shift, p):
    """p turned by theta about the up axis and moved by shift."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2]]) + shift


def join(target, found, joints, frames, times, k, fps):
    """Place the take onto the keys: (frames, times, start per frame for Target.pose, frames to write, far seam off).

    k: model frames per scene frame. The far seam's offset is (metres, degrees, gap seconds, travel metres).
    """
    before, after = found
    if before and before.seam - 1 < frames[0]:
        extra = np.arange(before.seam - 1, frames[0])
        frames, times = np.r_[extra, frames], np.r_[times[0] - (frames[0] - extra) * k, times]
    if after and after.seam + 1 > frames[-1]:
        extra = np.arange(frames[-1] + 1, after.seam + 2)
        frames, times = np.r_[frames, extra], np.r_[times, times[-1] + (extra - frames[-1]) * k]
    fits, at = [], len(joints) - (after.frames if after else 0)
    if before:
        fits.append(fit(before.joints[-FIT:], joints[before.frames - FIT : before.frames]))
    if after:
        fits.append(fit(after.joints[:FIT], joints[at : at + FIT]))
    (t0, s0), (t1, s1) = fits[0], fits[-1]
    w = np.zeros(len(frames))
    if before and after:
        w = np.clip((frames - before.seam) / (after.seam - before.seam), 0, 1)
    turn = (t1 - t0 + np.pi) % (2 * np.pi) - np.pi
    start = target.place(t0 + w * turn, s0 + w[:, None] * (s1 - s0))
    keep = (frames > before.seam if before else True) & (frames < after.seam if after else True)
    off = (0.0, 0.0, 0.0, 0.0)
    if before and after:
        p = joints[at, 0]
        d = turned(t1, s1, p) - turned(t0, s0, p)
        travel = np.linalg.norm(np.diff(joints[before.frames : at + 1, 0, [0, 2]], axis=0), axis=1).sum()
        seconds = (at - before.frames) / fps
        off = (float(np.hypot(d[0], d[2])), float(abs(np.degrees(turn))), float(seconds), float(travel))
    return frames, times, start, np.broadcast_to(keep, frames.shape), off


def note(off):
    """The warning for a far seam further off than the take can blend in, else ""."""
    d, turn, seconds, travel = off
    if d <= WARN * seconds + STRIDE * travel and turn <= WARN_TURN * seconds:
        return ""
    return f"The keys after are {d:.2f} m and {turn:.0f} deg from where this motion ends: blended in, feet may slide"


def ease(target, found, frames, basis, loc, fps):
    """Ease the take into the keys at each seam: the difference in pose and its change per frame fades over FADE."""
    for side, sign in zip(found, (1, -1), strict=True):
        if side is None:
            continue
        at, out = (int(np.flatnonzero(frames == f)[0]) for f in (side.seam, side.seam - sign))
        d = (frames - side.seam) * sign
        w = np.clip(1 - d / (FADE * fps), 0, 1)
        w = np.where(d < 0, 0, w * w * (3 - 2 * w))[:, None]
        rots, locs = side.pose
        for name, q in rots.items():
            if name in basis:
                a, b = (_rotation_vector(quat.qmul(q[k], quat.qinv(basis[name][i]))) for k, i in ((1, at), (0, out)))
                q = quat.qmul(_from_rotation_vector((a + (a - b) * d[:, None]) * w), basis[name])
                basis[name] = quat.continuous(q / np.linalg.norm(q, axis=1, keepdims=True))
        root = target.map[0]
        if loc is not None and root in locs:
            a, b = locs[root][1] - loc[at], locs[root][0] - loc[out]
            loc += (a + (a - b) * d[:, None]) * w


def _rotation_vector(q):
    """A rotation as its rotation vector (axis times angle)."""
    q = q * (1 if q[3] >= 0 else -1)
    sin = np.linalg.norm(q[:3])
    return q[:3] * (2 * np.arctan2(sin, q[3]) / sin) if sin > 1e-12 else 2 * q[:3]


def _from_rotation_vector(v):
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    axis = np.divide(v, angle, out=np.zeros_like(v), where=angle > 1e-12)
    return np.concatenate([axis * np.sin(angle / 2), np.cos(angle / 2)], -1)


def rigs():
    """The file's rigs whose takes join their keys."""
    return [o for o in generate.rigs() if any(b.anchor_key.strip("|") for b in blocks.settings(o).blocks)]


def check(obj):
    """Mark blocks dirty whose keys changed since last seen. After a load, undo or redo it only notes what is there."""
    m, s, uid = model.loaded(), blocks.settings(obj), obj.session_uid
    now = block_keys(s, plan(obj, m)) if m and s else {}
    seen, _state.seen[uid] = _state.seen.get(uid), now
    if seen is None or now == seen:
        return
    stale = [b.uid for b in s.blocks if b.uid in now and now[b.uid] != b.anchor_key] if s else []
    if stale:
        blocks.mark_dirty(obj, *stale, wait=True)


def _on_depsgraph(_scene, depsgraph):
    if any(isinstance(u.id, bpy.types.Action) for u in depsgraph.updates):
        for obj in rigs():
            generate.schedule(_state.due, obj.session_uid, DEBOUNCE, _tick)


def _tick():
    for uid in generate.pop_ready(_state.due):
        obj = generate.by_uid(uid)
        if obj is not None and obj.type == "ARMATURE":
            check(obj)
    return 0.05 if _state.due else None


def sync():
    """The depsgraph handler is there only while a rig in the file joins its keys."""
    generate.watch_depsgraph(_on_depsgraph, bool(rigs()))


def on_file():
    _state.due.clear()
    _state.seen.clear()
    sync()


def unregister():
    global _state
    generate.watch_depsgraph(_on_depsgraph, False)
    generate.stop_timer(_tick)
    _state = _State()
