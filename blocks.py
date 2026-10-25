"""Prompt blocks: text blocks on the timeline, in order and never overlapping, kept on the action the rig plays.
Blocks that touch form a run, made as one motion. A gap starts the next run afresh.
"""

import json
import math
import random

import bpy
import numpy as np

from . import continue_keys, generate, model, paths, retarget, timeline_motion

MIN_SECONDS, DEFAULT_SECONDS, MAX_SECONDS = 0.5, 3.0, 10.0
TOUCH = 2  # frames: blocks this close still touch
PALETTE = [
    (0.36, 0.55, 0.85),
    (0.89, 0.55, 0.30),
    (0.42, 0.72, 0.47),
    (0.80, 0.42, 0.55),
    (0.58, 0.47, 0.82),
    (0.85, 0.74, 0.32),
    (0.33, 0.70, 0.74),
    (0.72, 0.52, 0.40),
]


def stored(idb):
    """idb holds SprongMotion data (read without creating it)."""
    return "sprongmotion" in (idb.bl_system_properties_get() or {})


def settings(obj, create=False):
    """The timeline of the action obj plays, None without one. create: make an action or a local copy."""
    if obj is None or obj.type != "ARMATURE":
        return None
    ad = obj.animation_data
    action = ad.action if ad else None
    if create and (action is None or (not local(action) and not (stored(action) and len(action.sprongmotion.blocks)))):
        retarget.channelbag(obj)
        action = obj.animation_data.action
    if action is None or not (create or stored(action)):
        return None
    if create and not stored(action):
        action.sprongmotion.next_uid = 1  # a write: reading alone does not store the group
    return action.sprongmotion


def local(idb):
    return idb.library is None and idb.override_library is None


def players(action):
    """The file's local armatures playing action, the active object first."""
    found = [o for o in bpy.data.objects if o.type == "ARMATURE" and o.library is None and o.animation_data
             and o.animation_data.action == action]  # fmt: skip
    return sorted(found, key=lambda o: o != getattr(bpy.context, "active_object", None))


def player(idb):
    """The rig an edit of idb's data is for: the object itself, or for an action the rig playing it (None if none)."""
    return idb if isinstance(idb, bpy.types.Object) else next(iter(players(idb)), None)


def armature(context):
    obj = context.active_object
    return obj if obj is not None and obj.type == "ARMATURE" else None


def has_blocks(context):
    s = settings(context.active_object)
    return s is not None and len(s.blocks) > 0


def pick(context, index):
    """(active armature, block index): index if it names a block, else the active block."""
    obj = context.active_object
    s = settings(obj)
    return obj, index if 0 <= index < len(s.blocks) else s.active


def fps(scene=None):
    scene = scene or bpy.context.scene
    return scene.render.fps / scene.render.fps_base


def limits():
    """Block length in frames: at least MIN_SECONDS, at most the model's maximum less the seams to both neighbours."""
    f = fps()
    return round(MIN_SECONDS * f), int(model.max_seconds() * f)


def find(s, uid):
    return next((i for i, b in enumerate(s.blocks) if b.uid == uid), -1)


def has_motion(b):
    """The block is enabled and has text."""
    return bool(b.enabled and b.text.strip())


def timeline_blocks(s):
    """The blocks that make motion, in order."""
    return [b for b in s.blocks if has_motion(b)]


def runs(s):
    """The blocks in runs of touching ones. With Loop or a path the timeline is one run: one cycle, one curve."""
    out = []
    for b in s.blocks:
        if out and (s.loop or s.path or b.start - out[-1][-1].end <= TOUCH):
            out[-1].append(b)
        else:
            out.append([b])
    return out


def continues(s):
    """{uid: the enabled block with text it goes on from, 0 for none} over those blocks."""
    out = {}
    for run in runs(s):
        prev = 0
        for b in run:
            if b.enabled and b.text.strip():
                out[b.uid], prev = prev, b.uid
    return out


def chain(s, i):
    """The indices of the blocks touching on from block i."""
    k = i + 1
    while k < len(s.blocks) and s.blocks[k].start - s.blocks[k - 1].end <= TOUCH:
        k += 1
    return range(i + 1, k)


def shift(s, indices, frames):
    for k in indices:
        b = s.blocks[k]
        b.start, b.end = b.start + frames, b.end + frames


def push(s, i):
    """The blocks on either side of block i that overlap it move out of the way, and theirs in turn."""
    for k in range(i + 1, len(s.blocks)):
        shift(s, [k], max(0, s.blocks[k - 1].end - s.blocks[k].start))
    for k in range(i - 1, -1, -1):
        shift(s, [k], min(0, s.blocks[k + 1].start - s.blocks[k].end))


def place(s, i, start):
    """Block i moved to start, or to the nearest free place when that overlaps another block. Its new index."""
    b, n = s.blocks[i], s.blocks[i].end - s.blocks[i].start
    others = [x for k, x in enumerate(s.blocks) if k != i]
    edges = [-math.inf, *(e for x in others for e in (x.start, x.end)), math.inf]
    spots = [min(max(start, lo), hi - n) for lo, hi in zip(edges[::2], edges[1::2], strict=True) if hi - lo >= n]
    b.start = int(min(spots, key=lambda at: abs(at - start)))
    b.end = b.start + n
    j = sum(x.start < b.start for x in others)
    s.blocks.move(i, j)
    return j


def _new_block(s):
    b = s.blocks.add()
    b.uid, s.next_uid = s.next_uid, s.next_uid + 1
    b.color = PALETTE[(b.uid - 1) % len(PALETTE)]
    b.seed = random.randrange(2**31)
    b.in_place = len(s.blocks) > 1 and all(x.in_place for x in s.blocks[:-1])  # follows the rig-wide In Place
    return b


def add(obj, text, seconds=DEFAULT_SECONDS):
    s = settings(obj, create=True)
    with generate.hold():
        b = _new_block(s)
        i = len(s.blocks) - 1
        lo, hi = limits()
        b.start = s.blocks[i - 1].end if i else bpy.context.scene.frame_current
        b.end = b.start + max(lo, min(hi, round(seconds * fps())))
        b.text, s.active = text, i
        snap(obj, i)
        mark_dirty(obj, b.uid)
    return b


def snap(obj, index, frame=None):
    """A block added (or dropped at frame) in a gap between the rig's own keys fills it when the gap fits a block."""
    s, (lo, hi) = settings(obj), limits()
    b = s.blocks[index]
    g = continue_keys.gap(obj, b.start if frame is None else frame)
    if g is None:
        return
    start = g[0] if index == 0 else b.start
    if lo <= g[1] - start <= hi:
        b.start, b.end = start, g[1]
        push(s, index)


def fill(obj, lo, hi, text):
    """Blocks with text exactly over frames lo..hi, several if the gap is longer than one block can be."""
    s = settings(obj, create=True)
    n = max(1, math.ceil((hi - lo) / limits()[1]))
    with generate.hold():
        made = []
        for k in range(n):
            b = _new_block(s)
            b.start, b.end, b.text = lo + (hi - lo) * k // n, lo + (hi - lo) * (k + 1) // n, text
            made.append(b.uid)
            s.blocks.move(len(s.blocks) - 1, sum(x.start < b.start for x in s.blocks[:-1]))
        s.active, after = find(s, made[0]), find(s, made[-1]) + 1
        if after < len(s.blocks) and "made_fresh" in s.blocks[after]:
            s.blocks[after]["made_fresh"] = s.blocks[after].uid in timeline_motion.fresh_uids(s)
        mark_dirty(obj, *made)


def remove(obj, index):
    """The blocks touching on from it move up into its place."""
    s = settings(obj)
    after, n = chain(s, index), s.blocks[index].end - s.blocks[index].start
    s.blocks.remove(index)
    shift(s, [k - 1 for k in after], -n)
    s.active = min(s.active, len(s.blocks) - 1)
    if index < len(s.blocks):
        mark_dirty(obj, s.blocks[index].uid)
    else:
        generate.edited(obj)


def resize(obj, index, start=None, end=None, follow=False):
    """Resize a block, pushing neighbours it would overlap. follow: the blocks touching it move with the edge."""
    s = settings(obj)
    b, (lo, hi), after = s.blocks[index], limits(), chain(s, index)
    k = index
    while k > 0 and s.blocks[k].start - s.blocks[k - 1].end <= TOUCH:
        k -= 1
    if start is not None:
        was, b.start = b.start, max(b.end - hi, min(b.end - lo, start))
        if follow:
            shift(s, range(k, index), b.start - was)
    if end is not None:
        was, b.end = b.end, max(b.start + lo, min(b.start + hi, end))
        if follow:
            shift(s, after, b.end - was)
    paths.pin(obj, b)
    push(s, index)


def move(obj, index, to):
    """Block index to place to among the blocks from index to to: they keep their places' gaps between them."""
    s = settings(obj)
    a, z = sorted((index, to))
    gaps = [s.blocks[k + 1].start - s.blocks[k].end for k in range(a, z)] + [0]
    at = s.blocks[a].start
    s.blocks.move(index, to)
    for k in range(a, z + 1):
        b = s.blocks[k]
        b.start, b.end = at, at + (b.end - b.start)
        at = b.end + gaps[k - a]
    push(s, z)
    s.active = to
    mark_dirty(obj, *(s.blocks[k].uid for k in range(min(index, to), max(index, to) + 1)))


def split(obj, index, frame):
    """Split at frame. Each half keeps its part of the block's take, so the motion stays as it was."""
    s = settings(obj)
    lo, (start, end) = limits()[0], (s.blocks[index].start, s.blocks[index].end)
    if not start + lo <= frame <= end - lo:
        return None
    with generate.hold():
        _new_block(s)
        s.blocks.move(len(s.blocks) - 1, index + 1)
        b, n = s.blocks[index], s.blocks[index + 1]
        n.text, n.start, n.end, n.locked, n.enabled = b.text, frame, end, b.locked, b.enabled
        n["in_place"], n["back_to_start"], b["back_to_start"] = b.in_place, b.back_to_start, False  # the end walks back
        dirty, b.end, s.active = b.dirty, frame, index + 1
        if _split_takes(b, n, (frame - start) / (end - start)):
            b.dirty = n.dirty = dirty
            generate.edited(obj)
        else:
            b.variants = 0
            mark_dirty(obj, b.uid, n.uid)
    return s.blocks[index + 1]


def _split_takes(b, n, at):
    """Give n b's takes from the fraction at on. False without a take or when a half would be too short."""
    m = model.loaded()
    t = takes(b, m.id) if m else None
    k = round(at * b.tokens)
    if t is None or min(k, b.tokens - k) < math.ceil(MIN_SECONDS * m.fps / m.fpt):
        return False
    variant = b.variant
    store(b, t[:, :k], m.id)
    store(n, t[:, k:], m.id)
    n["variant"] = variant
    return True


def to_clip(b):
    """The block as a JSON string for the clipboard: its settings and its takes (codes are rig-independent)."""
    take = None
    if b.variants and b.get("codes") is not None:
        keys = ("codes", "model", "made_text", "made_fresh")
        take = {k: list(b[k]) if k == "codes" else b[k] for k in keys if k in b}
        take.update(variants=b.variants, tokens=b.tokens, variant=b.variant)
    return json.dumps({
        "sprongmotion": "block", "text": b.text, "seconds": (b.end - b.start) / fps(), "seed": b.seed,
        "color": list(b.color), "locked": b.locked, "fresh": b.fresh,
        "in_place": b.in_place, "back_to_start": b.back_to_start, "take": take,
    })  # fmt: skip


def _parse_clip(clip):
    try:
        d = json.loads(clip)
        if not isinstance(d, dict) or d.get("sprongmotion") != "block":
            return None
        t = d["take"]
        if t is not None:
            t = {**t, "codes": [int(c) for c in t["codes"]], "variants": int(t["variants"]), "tokens": int(t["tokens"])}
            if t["variants"] * t["tokens"] <= 0 or len(t["codes"]) % (t["variants"] * t["tokens"]):
                t = None
        return (str(d["text"]), float(d["seconds"]), int(d["seed"]), [float(c) for c in d["color"]][:3],
                bool(d["locked"]), bool(d["fresh"]), bool(d.get("in_place", False)),
                bool(d.get("back_to_start", False)), t)  # fmt: skip
    except (ValueError, KeyError, TypeError):
        return None


def paste(obj, clip, after):
    """Insert a block from to_clip's string after block `after` (-1: at the playhead). None if clip holds no block."""
    d = _parse_clip(clip)
    if d is None:
        return None
    text, seconds, seed, color, locked, fresh, in_place, back, t = d
    s, m, (lo, hi) = settings(obj, create=True), model.load(), limits()
    later = chain(s, after) if after >= 0 else range(0)
    with generate.hold():
        _new_block(s)
        i = after + 1
        s.blocks.move(len(s.blocks) - 1, i)
        b = s.blocks[i]
        b.start = s.blocks[after].end if after >= 0 else bpy.context.scene.frame_current
        b.end = b.start + max(lo, min(hi, round(seconds * fps())))
        b.text, b.seed, b.color, b.locked, b.fresh = text, max(0, seed) % 2**31, color, locked, fresh
        b["in_place"], b["back_to_start"] = in_place, back
        if t and m and t.get("model") == m.id:
            b["codes"], b["model"] = t["codes"], m.id
            b["made_text"], b["made_fresh"] = t.get("made_text", text.strip()), bool(t.get("made_fresh"))
            b.variants, b.tokens = t["variants"], t["tokens"]
            b["variant"] = min(max(0, int(t.get("variant", 0))), b.variants - 1)
            b.dirty = False
        else:
            b.dirty = True
        shift(s, [k + 1 for k in later], b.end - b.start)
        push(s, i)
        s.active = i
        generate.edited(obj)
    return s.blocks[i]


def mark_dirty(obj, *uids, wait=False):
    s = settings(obj)
    for uid in uids:
        i = find(s, uid)
        if i >= 0 and not s.blocks[i].locked:
            s.blocks[i].dirty = True
    generate.edited(obj, wait=wait)


def store(block, codes, model_id):
    """codes: int16 [variants, tokens, layers], kept in the .blend with the model id: reopening or undo only decodes."""
    block["codes"] = codes.reshape(-1).astype(np.int32).tolist()
    block["model"], block["made_text"] = model_id, block.text.strip()
    block["made_fresh"] = block.uid in timeline_motion.fresh_uids(block.id_data.sprongmotion)
    block.variants, block.tokens = codes.shape[0], codes.shape[1]
    block.remake = False
    if block.variant >= block.variants:
        block["variant"] = 0  # an ID write: the RNA update would count as a user edit
    block.dirty = False


def takes(block, model_id):
    """Every take's codes, int16 [variants, tokens, layers], or None if the block has none from this model."""
    flat = block.get("codes")
    if flat is None or not block.variants or block.get("model") != model_id:
        return None
    return np.array(flat, np.int16).reshape(block.variants, block.tokens, -1)


def codes(block, model_id, variant=None):
    t = takes(block, model_id)
    return None if t is None else t[min(block.variant if variant is None else variant, block.variants - 1)]


def redraw():
    wm = bpy.context.window_manager
    for win in wm.windows if wm else ():
        for area in win.screen.areas:
            if area.type in {"DOPESHEET_EDITOR", "VIEW_3D"}:
                area.tag_redraw()
