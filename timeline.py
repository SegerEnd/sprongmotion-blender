"""Generate calls over the timeline's blocks, and decoding the timeline run by run."""

import contextlib
import math
import random
from collections import namedtuple

import numpy as np

from . import anchor, blocks, model, paths, quat, return_to_start, takes

KEEP, MAKE, STUB = range(3)  # a block's codes kept, made, or made as a short stand-in whose take is not used
Part = namedtuple("Part", "uid text seed codes frames")  # a stretch of a block in a call: kept codes or frames to make


def uids(s):
    return [b.uid for b in blocks.timeline_blocks(s)]


def run_uids(s):
    """The timeline's uids run by run (blocks.runs): each run is made and decoded on its own."""
    line = set(uids(s))
    runs = ([b.uid for b in run if b.uid in line] for run in blocks.runs(s))
    return [run for run in runs if run]


def blend_uids(s):
    """The blocks Fresh Start makes alone and blends in: on, not the first of a run, no Loop."""
    if s.loop:
        return set()
    later = {uid for run in run_uids(s) for uid in run[1:]}
    return {b.uid for b in s.blocks if b.fresh and b.uid in later}


def fresh_uids(s):
    """The blocks made without the one before them: the first of each later run, and Fresh Start."""
    return {run[0] for run in run_uids(s)[1:]} | blend_uids(s)


def has_take(b, m):
    return blocks.has_motion(b) and blocks.codes(b, m.id) is not None


def needs_take(b, model_id):
    """The block makes motion and has no up-to-date take."""
    has = b.variants and b.get("model") == model_id
    fresh = b.uid in fresh_uids(b.id_data.sprongmotion)
    made_for = b.get("made_text", b.text.strip()), b.get("made_fresh", False)
    stale = made_for != (b.text.strip(), fresh) and not b.locked  # a take is for its own text and Fresh Start
    remade = b.dirty and (not b.locked or b.remake)
    return blocks.has_motion(b) and bool(not has or stale or remade)


def with_sides(segs, sides, line):
    """segs with the keys' context blocks (anchor.Side) before the timeline's first block and after its last."""
    before, after = sides
    head = [(before, KEEP)] if before and segs and segs[0][0].uid == line[0] else []
    ends_timeline = segs and segs[-1][1] != STUB and segs[-1][0].uid == line[-1]
    tail = [(after, KEEP)] if after and ends_timeline else []
    return head + segs + tail


def block_spans(segs, first, offset=0):
    """(block uid, first frame, end frame) per block in a call, without the context blocks."""
    pairs = zip(segs, first, first[1:], strict=False)
    return [(b.uid, offset + f0, offset + f1) for (b, _), f0, f1 in pairs if b.uid >= 0]


def arrays(motion):
    return [motion.rotations(), motion.joints(), motion.root(), motion.contacts(), motion.speed()]


def turn_about_up(rot, joints, root, angle):
    """Turn a motion about the up axis through the origin (in place)."""
    c, s = np.cos(angle), np.sin(angle)
    rot[:, 0] = quat.qmul(np.array([0, np.sin(angle / 2), 0, np.cos(angle / 2)]), rot[:, 0])
    for p in (joints, root):
        p[..., 0], p[..., 2] = c * p[..., 0] + s * p[..., 2], -s * p[..., 0] + c * p[..., 2]


def _continue_from_previous(previous, run):
    """Move a run (arrays) to where the run before it ends, turned to its facing."""
    rot, joints, root = run[:3]
    facing = [np.arctan2(*(j[1] - j[2])[[0, 2]]) for j in (previous[1][-1], joints[0])]  # from the hips
    turn_about_up(rot, joints, root, facing[0] - facing[1])
    joints[..., [0, 2]] += previous[1][-1, 0, [0, 2]] - joints[0, 0, [0, 2]]
    root[:, [0, 2]] += previous[2][-1, [0, 2]] - root[0, [0, 2]]


def without_travel(s, motion_arrays, spans):
    """Rotations, joints, contacts and speed, without the root's horizontal travel in blocks in place."""
    rot, joints, root, contacts, speed = motion_arrays
    step = np.diff(root[:, [0, 2]], axis=0, prepend=0)
    in_place = {b.uid for b in s.blocks if b.in_place}
    still = np.zeros(len(step), bool)
    for uid, f0, f1 in spans:
        still[f0:f1] |= uid in in_place
    if still.any():
        still[0] = False  # no step into the first frame
        joints[..., [0, 2]] -= np.cumsum(step * still[:, None], axis=0)[:, None]
    return rot, joints, contacts, speed


def _seconds(m, b, kind, codes):
    """The length a block gets in a call, in seconds."""
    if kind == KEEP:
        frames = m.fpt * len(codes)
    elif isinstance(b, tuple):
        frames = b.frames
    elif kind == MAKE:
        frames = (b.end - b.start) / blocks.fps() * m.fps
    else:
        frames = 0
    seconds = round(frames / m.fpt) * m.fpt / m.fps
    return max(blocks.MIN_SECONDS, min(model.max_seconds(), seconds))


def run_blocks(m, obj, segs, variant=0, made=None, path=True, in_place=False):
    """Model.blocks for [(block, KEEP | MAKE | STUB)]. path: True for the rig's path, False for none, or points."""
    s, seconds, keep = blocks.settings(obj), [], []
    for b, kind in segs:
        codes = None
        if kind == KEEP and isinstance(b, tuple):
            codes = b.codes
        elif kind == KEEP:
            codes = (made or {}).get(b.uid, blocks.codes(b, m.id))
        seconds.append(_seconds(m, b, kind, codes))
        keep.append(codes)
    seeds = [(b.seed + variant) % 2**31 for b, _ in segs]
    texts = [b.text for b, _ in segs]
    if path is True:
        points = paths.points(obj)
    elif path is False:
        points = None
    else:
        points = path
    return m.blocks(texts, seconds, seeds, keep, obj.sprongmotion.foot_lock, in_place, s.loop, points)


def _range_parts(b, codes, k0, k1, m):
    """The block's codes as kept, made and kept parts, made from token k0 to k1."""
    parts = [(Part(b.uid, b.text, 0, codes[:k0], 0), KEEP)] if k0 else []
    parts.append((Part(b.uid, b.text, random.randrange(2**31), None, (k1 - k0) * m.fpt), MAKE))
    if k1 < b.tokens:
        parts.append((Part(b.uid, b.text, 0, codes[k1:], 0), KEEP))
    return parts


def remake_range(obj, index, f0, f1):
    """Make frames f0..f1 of the block again and keep its other codes exactly. None when done, else why not."""
    m, s = model.load(), blocks.settings(obj)
    b = s.blocks[index]
    all_takes = blocks.takes(b, m.id) if m else None
    line = blocks.timeline_blocks(s)
    if all_takes is None or any(needs_take(x, m.id) for x in line):
        return model.error() if m is None else "Make the takes first"
    variant, n = min(b.variant, len(all_takes) - 1), b.tokens
    codes, least = all_takes[variant], math.ceil(blocks.MIN_SECONDS * m.fps / m.fpt)
    k0, k1 = (round((min(max(f, b.start), b.end) - b.start) / (b.end - b.start) * n) for f in (f0, f1))
    k0 = 0 if k0 < least else k0  # a kept part too short to stand alone is made too
    k1 = n if n - k1 < least else k1
    if k1 - k0 < least:
        return f"Pick at least {blocks.MIN_SECONDS:g} s"
    parts = _range_parts(b, codes, k0, k1, m)
    run = next(r for r in run_uids(s) if b.uid in r)
    segs = [seg for x in line if x.uid in run for seg in (parts if x.uid == b.uid else [(x, KEEP)])]
    segs = segs[run.index(b.uid) :] if b.uid in fresh_uids(s) else segs
    segs = with_sides(segs, anchor.sides(obj, m), [x.uid for x in line])
    with run_blocks(m, obj, segs) as (motion, first):
        at = first[next(i for i, (_, kind) in enumerate(segs) if kind == MAKE)] // m.fpt
        made = motion.codes()[at : at + k1 - k0]
    all_takes[variant] = np.concatenate([codes[:k0], made, codes[k1:]])
    blocks.store(b, all_takes, m.id)
    takes.apply(obj, push=False)
    return None


@contextlib.contextmanager
def whole(obj, path=True, sides=True, in_place=False):
    """Yields (motion, [(block, first frame, end frame)]) of the timeline, or (None, [])."""
    m, s = model.load(), blocks.settings(obj)
    segs = [(b, KEEP) for b in s.blocks if m and has_take(b, m)] if s else []
    if not segs:
        yield None, []
        return
    segs = with_sides(segs, anchor.sides(obj, m), uids(s)) if sides else segs
    with run_blocks(m, obj, segs, path=path, in_place=in_place) as (motion, first):
        pairs = zip(segs, first, first[1:], strict=False)
        yield motion, [(b, f0, f1) for (b, _), f0, f1 in pairs if b.uid >= 0]


def decode(obj, fits=None):
    """(rotations, joints, contacts, speed, spans) of the timeline decoded run by run, or None without a take."""
    m, s = model.loaded(), blocks.settings(obj)
    sides, line, decoded, spans, at = anchor.sides(obj, m), uids(s), [], [], 0
    for run in run_uids(s):
        segs = [(b, KEEP) for b in s.blocks if b.uid in run and has_take(b, m)]
        if not segs:
            continue
        segs = with_sides(segs, sides, line)
        with run_blocks(m, obj, segs) as (motion, first):
            run_arrays = arrays(motion)
        return_to_start.walk_back(obj, m, segs, first, run_arrays, fits)
        if decoded:
            _continue_from_previous(decoded[-1], run_arrays)
        spans += block_spans(segs, first, at)
        decoded.append(run_arrays)
        at += len(run_arrays[0])
    if not decoded:
        return None
    joined = [np.concatenate(a) for a in zip(*decoded, strict=True)]
    return *without_travel(s, joined, spans), spans
