"""Decoded takes onto the rig: timing, retargeting, foot and hand contacts, easing to rest and writing the keys."""

import bpy
import numpy as np

from . import blocks, continue_keys, generate, keyframes, model, paths, quat, retarget, rig, rig_data, timeline_motion

FRESH_SECONDS = 0.4  # a Fresh Start block blends in from the previous block's last pose


def _eases_to_rest(settings):
    p = model.prefs()
    return p is not None and p.after_block == "REST" and not settings.loop  # a loop closes on itself


def timing(segs):
    """Scene frames and the model frame each one samples: every block's take stretched over its own frame range."""
    frames, times = [], []
    for k, (b, first, end) in enumerate(segs):
        next_start = segs[k + 1][0].start if k + 1 < len(segs) else None
        touch = next_start is not None and next_start - b.end <= blocks.TOUCH
        stop = next_start if touch else b.end  # up to the next block when they touch
        f = np.arange(b.start, stop + (not touch))
        frames.append(f)
        t = first + (f - b.start) / (stop - b.start) * (end - first)
        before_gap = not touch and next_start is not None
        times.append(np.minimum(t, end - 1) if before_gap else t)  # the end frame shows the block's own last frame
    return np.concatenate(frames), np.concatenate(times)


def apply(obj, push, decoded=None):
    """Key the rig from its takes. decoded: what timeline_motion.decode returns, when the last generate call made it."""
    decoded = decoded or timeline_motion.decode(obj)
    if decoded is None:
        return
    rot, joints, contacts, speed, spans = decoded
    s, m = blocks.settings(obj), model.loaded()
    _fresh_starts(s, rot, joints, spans, m.fps)
    segs = [(s.blocks[blocks.find(s, uid)], f0, f1) for uid, f0, f1 in spans]
    frames, times = timing(segs)
    line, (before, after) = timeline_motion.uids(s), continue_keys.sides(obj, m)
    joined = before if spans[0][0] == line[0] else None, after if spans[-1][0] == line[-1] else None
    starts = [b.start for b, *_ in segs]
    off = write_take_keys(obj, rot, joints, contacts, frames, times, starts, sides=joined)
    for b in s.blocks:
        note = continue_keys.note(off) if joined[1] and b.uid == line[-1] else ""
        if b.seam_note != note:
            b.seam_note = note
    continue_keys.sync()
    if s.path or any(b.path_note for b in s.blocks):
        paths.noted(obj, speed, spans)
    if push:
        generate.undo_push("SprongMotion take")


def _fresh_starts(s, rot, joints, spans, fps):
    """A Fresh Start block starts on from the previous block's last pose and speed; the difference fades out."""
    fresh = timeline_motion.blend_uids(s)
    for uid, f0, f1 in spans:
        if uid not in fresh or f0 <= 0:
            continue
        n = min(f1 - f0, round(FRESH_SECONDS * fps))
        x = np.arange(n) / n
        left = (1 - x * x * (3 - 2 * x))[:, None, None]  # smoothstep from 1 to 0
        last, before_last = f0 - 1, max(f0 - 2, 0)
        last_turn = quat.qmul(rot[last], quat.qinv(rot[before_last]))
        ahead = quat.qmul(last_turn, rot[last])
        turn = quat.qmul(ahead, quat.qinv(rot[f0]))
        rot[f0 : f0 + n] = quat.qmul(quat.qpow(turn, left), rot[f0 : f0 + n])
        joints[f0 : f0 + n] += (2 * joints[last, 0] - joints[before_last, 0] - joints[f0, 0]) * left


def _contacts(s, target, frozen, joints, contacts, times):
    """Per frame: the foot contacts, and the feet and hands IK holds ([F, 2], None when off); not on frozen limbs."""
    at = np.round(times).astype(int).clip(0, len(contacts) - 1)
    on = contacts[at]
    planted = on[:, [0, 2]].copy() if s.foot_lock else None  # the ankle flags: on the toes alone the ankle rolls
    hands = retarget.hand_contacts(joints, model.loaded().fps)[at] if s.hand_lock else None
    for flags, limbs in ((planted, rig.LEGS), (hands, rig.ARMS)):
        for side, limb in enumerate(limbs):
            if flags is not None and frozen & {target.map.get(j) for j in limb}:
                flags[:, side] = False
    return on, planted, hands


def _held_bones(target, planted, hands):
    """{bone: frames it is held by IK}: those frames are keyed tighter, or planted feet slide."""
    held = {}
    for flags, limbs in ((planted, rig.LEGS), (hands, rig.ARMS)):
        if flags is None:
            continue
        for side in range(2):
            for j in limbs[side] + (0,):
                if j in target.map:
                    bone = target.map[j]
                    held[bone] = held.get(bone, False) | flags[:, side]
    return held


def _keep_frames(values, keep):
    return values[keep] if values is not None else None


def write_take_keys(obj, rot, joints, contacts, frames, times, borders, action=None, start=True, kind=None,
                    sides=(None, None)):  # fmt: skip
    """Retarget a decoded take and key it on obj's action (or action); returns how far off the far seam is."""
    s, settings = obj.sprongmotion, blocks.settings(obj)
    rig.ensure_map(obj)
    rig.prepare(obj)
    m = model.loaded()
    target = retarget.Target(obj, m.offsets)
    off = (0.0,) * 4
    joins_keys = start and any(sides)
    if joins_keys:
        frames, times, first, keep, off = continue_keys.join(
            target, sides, joints, frames, times, m.fps / blocks.fps(), m.fps
        )
    else:
        first = retarget.start_pose(obj, target, frames[0]) if start else None
        keep = np.ones(len(frames), bool)
    frozen = rig_data.frozen(obj)
    on, planted, hands = _contacts(s, target, frozen, joints, contacts, times)
    rot, root = retarget.resample(rot, times, quat=True), retarget.resample(joints[:, 0], times)
    basis, loc = target.pose(rot, root, planted, first, hands, s.floor_clamp)
    if joins_keys:
        continue_keys.ease(target, sides, frames, basis, loc, blocks.fps())
    frames, on, basis = frames[keep], on[keep], {name: q[keep] for name, q in basis.items()}
    loc, planted, hands = _keep_frames(loc, keep), _keep_frames(planted, keep), _keep_frames(hands, keep)
    contact_changes = frames[1:][(on[1:] != on[:-1]).any(1)]
    marks = np.r_[contact_changes, [b for b in borders if frames[0] <= b <= frames[-1]]]
    held = _held_bones(target, planted, hands)
    basis = {name: q for name, q in basis.items() if name not in frozen}
    root_bone = target.map.get(0)
    loc = None if root_bone in frozen else loc
    has_gaps = (np.diff(frames) > 1).any()
    if action is None and start and not settings.loop and (_eases_to_rest(settings) or has_gaps):
        frames, basis, loc, marks, held = _ease_to_rest(obj, target, settings, frames, basis, loc, marks, held)
    fit = None
    if s.key_density == "UNBAKED":
        fit = (s.key_error, s.key_error, marks, s.key_frames == "SHARED", held)
    retarget.write(obj, frames, basis, loc, root_bone, fit, action, kind)
    return off


def _ease_to_rest(obj, target, settings, frames, basis, loc, marks, held):
    """Where nothing follows a stretch of the take: the bones ease to rest (or hold the pose in a gap)."""
    root_bone = target.map.get(0)
    own = continue_keys.own_frames(obj, set(basis) | {root_bone})
    p = model.prefs()
    n = max(1, round((p.ease_seconds if p else 0.4) * blocks.fps()))
    up = quat.qrot(quat.qinv(target.rest[root_bone]), target.axes[:, 1]) if loc is not None else None
    taken = frames
    frames, basis, loc, more = to_rest(frames, basis, loc, own, n, not _eases_to_rest(settings), up)
    held = {bone: np.isin(frames, taken[on]) for bone, on in held.items()}
    return frames, basis, loc, np.r_[marks, more], held


def action_names(s):
    """A name per block that gets its own action: what block_actions pairs names with."""
    return [b.text.strip().capitalize()[:60] for b in blocks.timeline_blocks(s)]


def block_actions(obj, names=None, asset=False):
    """Each block with a take as its own action from frame 1, decoded alone; the rig keeps its own action."""
    m, s, made = model.load(), blocks.settings(obj), []
    named = zip(blocks.timeline_blocks(s), names or action_names(s), strict=False)
    path_parts = paths.parts(obj) if not s.in_place else {}
    for b, name in named:
        if not timeline_motion.has_take(b, m):
            continue
        path = not b.in_place and path_parts.get(b.uid, False)
        with timeline_motion.run_blocks(m, obj, [(b, timeline_motion.KEEP)], path=path, in_place=b.in_place) as (
            motion,
            first,
        ):
            rot, joints, contacts = motion.rotations(), motion.joints(), motion.contacts()
        frames = np.arange(1, b.end - b.start + 2)
        times = (frames - 1) / (frames[-1] - 1) * (first[1] - 1)
        action = bpy.data.actions.new(name)
        write_take_keys(obj, rot, joints, contacts, frames, times, [], action, start=False, kind=keyframes.KEYFRAME)
        action.frame_range, action.use_frame_range, action.use_cyclic = (1, frames[-1]), True, s.loop
        action.use_fake_user = True  # unassigned until a rig plays it: kept on save
        if asset:
            action.asset_mark()
        made.append(action)
    return made


def to_rest(frames, basis, loc, own, n, hold=False, up=None):
    """Ease to rest over n frames where no block and no own key (own) follows each stretch of the take.

    hold: keep the stretch's last pose in gaps instead. up: the root bone's local up axis, the only axis its location
    eases along. Returns (frames, basis, location, frames to key).
    """
    ends = np.r_[np.flatnonzero(np.diff(frames) > 1), len(frames) - 1]
    out, marks = [], []
    for i in ends:
        e = frames[i]
        s = frames[i + 1] if i + 1 < len(frames) else None  # the next stretch's start, None after the last
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
