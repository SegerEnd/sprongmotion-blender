"""Generation on the main thread. Small edits generate at once, so an edit and its take are one undo step; bigger jobs
run from a bpy.app.timers callback, one block's take per tick (one sm_ctx_generate_blocks call, ~0.2 s), so the UI
stays live and Cancel acts between blocks."""

import contextlib
import ctypes as C
import math
import os
import platform
import random
import time
import traceback
from collections import namedtuple

import bpy
import numpy as np

from . import anchor, assist, blocks, data, keys, paths, retarget, rig, sm

DEBOUNCE = 0.25  # seconds of quiet after an edit before a timed regenerate
EDGE_TOKENS = 12  # tokens sm_ctx_generate_blocks adds to a made block: seam 3 + look-ahead 8, a lone loop 12
FRESH_SECONDS = 0.4  # a Fresh Start block blends in from the previous block's last pose
SYNC_BLOCKS = 3  # single-take blocks generated inside the edit itself (~0.1 s each)
THREADS = 0  # model threads, 0 = every core
HERE = os.path.dirname(__file__)
MODEL = os.path.join(HERE, "model", "SprongMotion-14M-v1.0-Q8_0.gguf")
ARCH, EXT = {"arm64": "arm64", "aarch64": "arm64"}, {"Darwin": ".dylib", "Windows": ".dll"}


class _State:
    def __init__(self):
        self.lib = self.model = self.job = self.paths = self.help = None
        self.error = ""
        self.due, self.edits, self.words, self.near = {}, {}, {}, {}  # due and edits by the rig's session_uid
        self.hold, self.held = 0, set()
        self.settle = None  # "undo" or "redo": the step to take next (settle)


def by_uid(uid):
    """The armature with this session_uid (kept across undo and renames), or None."""
    return next((o for o in bpy.data.objects if o.session_uid == uid), None)


_s = _State()


def prefs():
    addon = bpy.context.preferences.addons.get(__package__)
    return addon.preferences if addon else None


def _eases(s):
    p = prefs()
    return p is not None and p.after_block == "REST" and not s.loop  # a loop closes on itself


def rigs():
    """The file's local armatures playing an action with SprongMotion blocks (read without creating any)."""
    return [o for o in bpy.data.objects if o.library is None and blocks.settings(o) is not None]


def reapply():
    """Every rig's takes keyed again from their codes (a preference changed), as one undo step: nothing is made."""
    if model() is None:
        return
    found = [o for o in rigs() if len(blocks.settings(o).blocks) and not pending(o)]
    for obj in found:
        apply(obj, push=False)
    if found:
        undo_push("SprongMotion: After a Block")


def bundled():
    """The library and model shipped inside the extension (lib/<platform>/, model/)."""
    system = platform.system()
    tag = f"{system.lower().replace('darwin', 'macos')}-{ARCH.get(platform.machine().lower(), 'x64')}"
    lib = os.path.join(HERE, "lib", tag, "libsprongmotion" + EXT.get(system, ".so"))
    return (lib if os.path.isfile(lib) else ""), (MODEL if os.path.isfile(MODEL) else "")


def model():
    """(Re)loads when the paths in the preferences changed: call it from operators and updates, never from a draw."""
    p, (lib, gguf) = prefs(), bundled()
    paths = (bpy.path.abspath(p.library_path) or lib, bpy.path.abspath(p.model_path) or gguf) if p else ("", "")
    if paths != _s.paths:
        unload()
        _s.paths, _s.error = paths, _load(*paths)
    return _s.model


def _load(lib_path, model_path):
    if not lib_path:
        return "Choose the SprongMotion library (libsprongmotion) in the add-on preferences"
    if not model_path:
        return "Choose a SprongMotion model file (.gguf) in the add-on preferences"
    if _s.lib and _s.lib.path != lib_path:
        return "Restart Blender to use another SprongMotion library"  # a loaded library cannot be unloaded safely
    try:
        _s.lib = _s.lib or sm.Lib(lib_path)
    except (OSError, AttributeError, sm.Error) as e:
        return f"Cannot load the library {lib_path}: {e}"
    try:
        _s.model = sm.Model(_s.lib, model_path, C.byref(sm.Hooks(THREADS)))
    except sm.Error as e:
        return f"Cannot load the model {model_path}: {e}"
    m = _s.model
    said = assist.lines("sentences.txt")
    words = set(m.vocab()) & set(assist.lines("words.txt")) | {w for s in said for w in s.split() if len(w) > 3}
    _s.help = assist.Help(said, words, m.vector)
    return ""


def loaded():
    """The model as loaded now, for draws: loads nothing."""
    return _s.model


def error():
    return _s.error


def unload():
    cancel()
    if _s.model:
        _s.model.free()
    _s.model, _s.paths, _s.words, _s.near, _s.help = None, None, {}, {}, None


def shutdown():
    unload()
    for fn in (_tick, _settle):
        if bpy.app.timers.is_registered(fn):
            bpy.app.timers.unregister(fn)


def max_seconds():
    """The longest block the runtime makes with whatever edges it has: the one limit (blocks.limits)."""
    m = _s.model
    return (m.max_seconds * m.fps / m.fpt - EDGE_TOKENS) * m.fpt / m.fps if m else blocks.MAX_SECONDS - 1.2


def unknown_words(text, look=True):
    """look: run the text encoder if text is new, else only read the cache (draws)."""
    if look and text not in _s.words and _s.model:
        try:
            _s.words[text] = _s.model.unknown_words(text)
        except sm.Error:
            _s.words[text] = []
        for w in _s.words[text]:
            if w not in _s.near and _s.help:
                _s.near[w] = _s.help.near(w)
    return _s.words.get(text, [])


def nearest(word):
    """Known words spelled like an unknown one, as unknown_words found them (reads only)."""
    return _s.near.get(word, [])


def helper():
    """The typing help, None without a model or when the preferences turn it off (reads only)."""
    p = prefs()
    return _s.help if p is None or p.typing_help else None


@contextlib.contextmanager
def hold():
    """Edits inside count as one: a single regenerate when the outermost hold ends."""
    _s.hold += 1
    try:
        yield
    finally:
        _s.hold -= 1
    if not _s.hold:  # not reached when the edits raised: nothing regenerates then
        uids, _s.held = _s.held, set()
        for obj in filter(None, map(by_uid, uids)):
            edited(obj)


def edited(obj, wait=False, force=False):
    """obj's blocks changed. Small jobs run now, so the edit and its take share one undo step; big ones, and slider
    drags (wait), run from the timer with their own undo step."""
    uid = obj.session_uid
    _s.edits[uid] = _s.edits.get(uid, 0) + 1
    if _s.hold:
        _s.held.add(uid)
        return
    r, s, m = obj.sprongmotion, blocks.settings(obj), model()
    if not (force or r.auto) or m is None or s is None:
        return
    if busy(obj):
        cancel()
    small = r.variants == 1 and sum(needs_take(b, m.id) for b in s.blocks) <= SYNC_BLOCKS
    if small and not wait and not busy():
        return run_now(obj)
    s.take_due = True  # in the edit's own undo step: its take comes in a later one (settle)
    _s.due[uid] = time.perf_counter() + DEBOUNCE
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=DEBOUNCE)


def run_now(obj):
    cancel()
    job = _Job(obj.session_uid, push=False, per_block=False)
    while not _drive(job):
        pass
    blocks.redraw()


def busy(obj=None):
    return _s.job is not None and (obj is None or _s.job.rig == obj.session_uid)


def pending(obj):
    """A take for obj is being made or waits for the timer."""
    return busy(obj) or obj.session_uid in _s.due


Progress = namedtuple("Progress", "rig uid done total")  # rig: the armature's session_uid; uid: the block being made


def progress():
    j = _s.job
    return Progress(j.rig, j.units[j.step][0][0], j.step, len(j.units)) if j and j.step < len(j.units) else None


def cancel():
    if _s.job:
        _s.job.stop()
    _s.job = None
    _s.due.clear()


def _drive(job):
    """Advance a job; any failure ends it with one message (and one console traceback), never a stuck job or a timer
    that raises every tick."""
    try:
        return job.advance()
    except Exception as e:
        traceback.print_exc()
        job.stop()
        _s.error = f"SprongMotion: {e}"
        return True


def _tick():
    now = time.perf_counter()
    if _s.job is None:
        ready = next((n for n, t in _s.due.items() if t <= now), None)
        if ready:
            del _s.due[ready]
            _s.job = _Job(ready, push=True)
    if _s.job is not None:
        if _drive(_s.job):
            _s.job = None
        blocks.redraw()
    return 0.01 if _s.job or _s.due else None


class _Job:
    """Makes the takes of every block that needs one, one sm_ctx_generate_blocks call per advance: first take 0 of the
    blocks left to right (per_block: one block per call, so a timer tick stays short; else all in one call), then
    each further take of each block against the first takes, so any take fits its neighbours. An edit meanwhile
    starts it over."""

    def __init__(self, uid, push, per_block=True):
        self.rig, self.push, self.per_block = uid, push, per_block
        self.action = _action(by_uid(uid))
        self.edits = object()  # never equals an edit count, so the first advance builds the unit list
        self.stop()

    def stop(self):
        self.todo, self.units, self.takes, self.step, self.decoded = [], [], {}, 0, None

    def advance(self):
        """One unit, or the end: store the takes and key the armature. True when the job is over."""
        obj, m = by_uid(self.rig), model()
        if obj is not None and (why := data.why_not_editable(obj)):
            raise RuntimeError(why)
        s = blocks.settings(obj) if obj is not None else None
        if s is None or m is None or _action(obj) != self.action:
            return True  # also after an action switch: the takes made so far are for the other action's blocks
        if self.edits != _s.edits.get(self.rig):
            self.plan(obj, m)
        if self.step < len(self.units):
            self.take(obj, m, *self.units[self.step])
            self.step += 1
            return False
        for uid, takes in self.takes.items():
            if (i := blocks.find(s, uid)) >= 0:
                blocks.store(s.blocks[i], np.stack(takes), m.id)
                s.blocks[i].anchor_key = self.joins.get(uid, "|")
        if paths.refit(obj):
            self.edits = object()  # the resized blocks are made in this same job, so in the edit's undo step
            return False
        for b in s.blocks:
            unknown_words(b.text)
        if s.take_due:
            s.take_due = False
        apply(obj, self.push, self.decoded)
        _apply_others(obj)
        _s.error = ""
        return True

    def plan(self, obj, m):
        """The units to make, after an edit: blocks whose keys around them changed (anchor) need a take too."""
        s = blocks.settings(obj)
        self.edits, self.takes, self.step, self.decoded = _s.edits.get(self.rig), {}, 0, None
        if len(obj.sprongmotion.bones) != len(rig.JOINTS):
            rig.fill_map(obj)
        self.sides = anchor.sides(obj, m)
        self.joins = anchor.block_keys(s, self.sides)
        for b in s.blocks:
            if b.anchor_key != self.joins.get(b.uid, "|"):
                b.dirty = True
        self.todo, fresh, first = [b.uid for b in s.blocks if needs_take(b, m.id)], fresh_uids(s), []
        run = {u: k for k, r in enumerate(_runs(s)) for u in r}
        for u in self.todo:
            if self.per_block or not first or u in fresh or run[u] != run[first[-1][0][-1]]:
                first.append(((u,), 0))  # a fresh block starts a call: no block before it in it
            else:
                first[-1] = ((*first[-1][0], u), 0)
        self.units = first + [((u,), v) for v in range(1, obj.sprongmotion.variants) for u in self.todo]

    def take(self, obj, m, make, variant):
        """The takes of the blocks in make, in a call with the blocks already final. A block still to be made after one
        goes in as a short stand-in, and with Loop also the last block: the runtime then gives each made block the same
        edges as in one call making them all (no pin, and its lookahead past the end towards a block made later)."""
        s = blocks.settings(obj)
        firsts, segs, last = {u: t[0] for u, t in self.takes.items()}, [], _timeline_uids(s)[-1]
        later = {b.uid for b in s.blocks if b.uid in self.todo and b.uid not in firsts and b.uid not in make}
        run = next(r for r in _runs(s) if make[0] in r)
        for b in s.blocks:
            if b.uid not in run:
                continue
            if b.uid in make or b.uid in firsts or (b.uid not in self.todo and _keeps(b, m)):
                segs.append((b, MAKE if b.uid in make else KEEP))
            elif b.uid in later and ((segs and segs[-1][1] == MAKE) or (s.loop and b.uid == last)):
                segs.append((b, STUB))
        if make[0] in fresh_uids(s):
            segs = segs[[b.uid for b, _ in segs].index(make[0]) :]
        whole = obj.sprongmotion.variants == 1 and [b.uid for b, _ in segs] == _timeline_uids(s)
        whole = whole and not any(paths.walks_back(s, b) for b in s.blocks)
        segs = _with_sides(segs, self.sides, _timeline_uids(s))
        with _blocks(m, obj, segs, variant, firsts) as (motion, first):
            codes = motion.codes()
            spans = [(b.uid, f0, f1) for (b, _), f0, f1 in zip(segs, first, first[1:], strict=False) if b.uid >= 0]
            self.decoded = (*_decoded(s, _arrays(motion), spans), spans) if whole else None
        for (b, kind), f0, f1 in zip(segs, first, first[1:], strict=False):
            if kind == MAKE:
                self.takes.setdefault(b.uid, []).append(codes[f0 // m.fpt : f1 // m.fpt])


def _action(obj):
    ad = obj.animation_data if obj else None
    return ad.action.session_uid if ad and ad.action else None


def _apply_others(obj):
    """The blocks are the action's: every other rig playing it keys them with its own bone map."""
    for other in blocks.players(obj.animation_data.action):
        if other != obj and not pending(other):
            apply(other, push=False)


def _timeline_uids(s):
    return [b.uid for b in s.blocks if b.enabled and b.text.strip()]


def _runs(s):
    """The timeline's uids run by run (blocks.runs): each run is made and decoded on its own."""
    line = set(_timeline_uids(s))
    return [r for r in ([b.uid for b in run if b.uid in line] for run in blocks.runs(s)) if r]


def blend_uids(s):
    """The blocks Fresh Start makes alone and blends in: on, not the first of a run, no Loop."""
    later = {u for r in _runs(s) for u in r[1:]}
    return set() if s.loop else {b.uid for b in s.blocks if b.fresh and b.uid in later}


def fresh_uids(s):
    """The blocks made without the one before them: the first of each run after the first, and Fresh Start."""
    return {r[0] for r in _runs(s)[1:]} | blend_uids(s)


def _keeps(b, m):
    return b.enabled and b.text.strip() and blocks.codes(b, m.id) is not None


KEEP, MAKE, STUB = range(3)  # a block's codes as they are, made, or a short made stand-in whose take is not used


def _with_sides(segs, found, line):
    """segs with the keys' context blocks (anchor.Side) before the timeline's first block and after its last."""
    before, after = found
    head = [(before, KEEP)] if before and segs and segs[0][0].uid == line[0] else []
    tail = [(after, KEEP)] if after and segs and segs[-1][1] != STUB and segs[-1][0].uid == line[-1] else []
    return head + segs + tail


def _arrays(motion):
    return [motion.rotations(), motion.joints(), motion.root(), motion.contacts(), motion.speed()]


def _go_on(prev, got):
    """A run after a gap starts where the run before it ends (_arrays of both): moved there, turned to its facing."""
    rot, joints, root = got[:3]
    a = np.subtract(*[np.arctan2(*(j[1] - j[2])[[0, 2]]) for j in (prev[1][-1], joints[0])])
    c, s = np.cos(a), np.sin(a)
    rot[:, 0] = retarget.qmul(np.array([0, np.sin(a / 2), 0, np.cos(a / 2)]), rot[:, 0])
    for p in (joints, root):
        p[..., 0], p[..., 2] = c * p[..., 0] + s * p[..., 2], -s * p[..., 0] + c * p[..., 2]
    joints[..., [0, 2]] += prev[1][-1, 0, [0, 2]] - joints[0, 0, [0, 2]]
    root[:, [0, 2]] += prev[2][-1, [0, 2]] - root[0, [0, 2]]


def _decoded(s, arrays, spans):
    """Rotations, joints, contacts and speed from _arrays; the joints without the root's horizontal travel in the
    frames of blocks in place, and moved by what they left out from there on, so the next block goes on from where
    they stand."""
    rot, joints, root, contacts, speed = arrays
    d = np.diff(root[:, [0, 2]], axis=0, prepend=0)
    held = {b.uid for b in s.blocks if b.in_place}
    still = np.zeros(len(d), bool)
    for uid, f0, f1 in spans:
        still[f0:f1] |= uid in held
    if still.any():
        still[0] = False  # no step into the first frame
        joints[..., [0, 2]] -= np.cumsum(d * still[:, None], axis=0)[:, None]
    return rot, joints, contacts, speed


def _blocks(m, obj, segs, variant=0, takes=None, path=True, in_place=False):
    """Model.blocks for [(block, KEEP | MAKE | STUB)]: made blocks get the length of their frames, kept ones their
    codes (or the take in takes this job just made). path: True = the rig's path if it has one, False = none, or
    the points themselves."""
    s, seconds, keep = blocks.settings(obj), [], []
    for b, kind in segs:
        part = isinstance(b, tuple)
        codes = None if kind != KEEP else b.codes if part else (takes or {}).get(b.uid, blocks.codes(b, m.id))
        frames = (m.fpt * len(codes) if kind == KEEP else b.frames if part
                  else (b.end - b.start) / blocks.fps() * m.fps if kind == MAKE else 0)  # fmt: skip
        seconds.append(max(blocks.MIN_SECONDS, min(max_seconds(), round(frames / m.fpt) * m.fpt / m.fps)))
        keep.append(codes)
    seeds = [(b.seed + variant) % 2**31 for b, _ in segs]
    texts = [b.text for b, _ in segs]
    pts = paths.points(obj) if path is True else None if path is False else path
    return m.blocks(texts, seconds, seeds, keep, obj.sprongmotion.foot_lock, in_place, s.loop, pts)


Part = namedtuple("Part", "uid text seed codes frames")  # a stretch of a block in a call: kept codes, or frames to make


def remake_range(obj, index, f0, f1):
    """A new take of frames f0..f1 of the block, its other codes kept bit-exact: the block's codes are split at token
    boundaries into kept / made / kept parts and made in one call with the other blocks kept (a part outside too short
    to stand alone is made too). None when done, else why not."""
    m, s = model(), blocks.settings(obj)
    b = s.blocks[index]
    takes = blocks.takes(b, m.id) if m else None
    line = [x for x in s.blocks if x.enabled and x.text.strip()]
    if takes is None or any(needs_take(x, m.id) for x in line):
        return error() if m is None else "Make the takes first"
    codes, n, least = takes[min(b.variant, len(takes) - 1)], b.tokens, math.ceil(blocks.MIN_SECONDS * m.fps / m.fpt)
    k0, k1 = (round((min(max(f, b.start), b.end) - b.start) / (b.end - b.start) * n) for f in (f0, f1))
    k0, k1 = 0 if k0 < least else k0, n if n - k1 < least else k1
    if k1 - k0 < least:
        return f"Pick at least {blocks.MIN_SECONDS:g} s"
    parts = [(Part(b.uid, b.text, 0, codes[:k0], 0), KEEP)] if k0 else []
    parts.append((Part(b.uid, b.text, random.randrange(2**31), None, (k1 - k0) * m.fpt), MAKE))
    parts += [(Part(b.uid, b.text, 0, codes[k1:], 0), KEEP)] if k1 < n else []
    run = next(r for r in _runs(s) if b.uid in r)
    segs = [seg for x in line if x.uid in run for seg in (parts if x.uid == b.uid else [(x, KEEP)])]
    segs = segs[run.index(b.uid) :] if b.uid in fresh_uids(s) else segs
    segs = _with_sides(segs, anchor.sides(obj, m), [x.uid for x in line])
    with _blocks(m, obj, segs) as (motion, first):
        at = first[next(i for i, (_, kind) in enumerate(segs) if kind == MAKE)] // m.fpt
        made = motion.codes()[at : at + k1 - k0]
    takes[min(b.variant, len(takes) - 1)] = np.concatenate([codes[:k0], made, codes[k1:]])
    blocks.store(b, takes, m.id)
    apply(obj, push=False)
    return None


def needs_take(b, model_id):
    has = b.variants and b.get("model") == model_id
    fresh = b.uid in fresh_uids(b.id_data.sprongmotion)
    made = b.get("made_text", b.text.strip()), b.get("made_fresh", False)
    stale = made != (b.text.strip(), fresh) and not b.locked  # a take is for its own text and Fresh Start
    return bool(b.enabled and b.text.strip() and (not has or stale or (b.dirty and (not b.locked or b.remake))))


@contextlib.contextmanager
def timeline(obj, path=True, sides=True, in_place=False):
    """(motion, [(block, first frame, end frame)]), or (None, []). sides: with the keys' context blocks around it (in
    the motion, not in the list). in_place: the runtime's In Place over the whole motion."""
    m, s = model(), blocks.settings(obj)
    segs = [(b, KEEP) for b in s.blocks if m and _keeps(b, m)] if s else []
    if not segs:
        yield None, []
        return
    segs = _with_sides(segs, anchor.sides(obj, m), _timeline_uids(s)) if sides else segs
    with _blocks(m, obj, segs, path=path, in_place=in_place) as (motion, first):
        yield motion, [(b, f0, f1) for (b, _), f0, f1 in zip(segs, first, first[1:], strict=False) if b.uid >= 0]


def timing(segs):
    """Scene frames and the model frame each one samples: every block's take stretched onto its own frame range (up to
    the next block when they touch). Before a gap the block's end frame shows its own last frame."""
    frames, times = [], []
    for k, (b, first, end) in enumerate(segs):
        after = segs[k + 1][0].start if k + 1 < len(segs) else None
        touch = after is not None and after - b.end <= blocks.TOUCH
        stop = after if touch else b.end
        f = np.arange(b.start, stop + (not touch))
        frames.append(f)
        t = first + (f - b.start) / (stop - b.start) * (end - first)
        times.append(t if touch or after is None else np.minimum(t, end - 1))
    return np.concatenate(frames), np.concatenate(times)


def apply(obj, push, decoded=None):
    """decoded: (rotations, pelvis, contacts, speed, [(block uid, first frame, end frame)]) of the whole timeline,
    when the last generate call decoded it already."""
    decoded = decoded or _decode(obj)
    if decoded is None:
        return
    *data, speed, spans = decoded
    s = blocks.settings(obj)
    _fresh_starts(s, *data[:2], spans, _s.model.fps)
    segs = [(s.blocks[blocks.find(s, uid)], f0, f1) for uid, f0, f1 in spans]
    frames, times = timing(segs)
    line, (before, after) = _timeline_uids(s), anchor.sides(obj, _s.model)
    found = before if spans[0][0] == line[0] else None, after if spans[-1][0] == line[-1] else None
    off = _key(obj, *data, frames, times, [b.start for b, *_ in segs], sides=found)
    for b in s.blocks:
        if b.seam_note != (note := anchor.note(off) if found[1] and b.uid == line[-1] else ""):
            b.seam_note = note
    anchor.sync()
    if s.path or any(b.path_note for b in s.blocks):
        paths.noted(obj, speed, spans)
    if push:
        undo_push("SprongMotion take")


def _decode(obj, fits=None):
    """The timeline decoded run by run, each run going on from where the one before it ends (and its Return to Start
    blocks walking back, paths.walk_back with fits): what apply takes as decoded, or None without a take."""
    m, s = _s.model, blocks.settings(obj)
    sides, line, parts, spans, at = anchor.sides(obj, m), _timeline_uids(s), [], [], 0
    for run in _runs(s):
        segs = [(b, KEEP) for b in s.blocks if b.uid in run and _keeps(b, m)]
        if not segs:
            continue
        segs = _with_sides(segs, sides, line)
        with _blocks(m, obj, segs) as (motion, first):
            got = _arrays(motion)
        paths.walk_back(obj, m, segs, first, got, fits)
        if parts:
            _go_on(parts[-1], got)
        spans += [
            (b.uid, at + f0, at + f1) for (b, _), f0, f1 in zip(segs, first, first[1:], strict=False) if b.uid >= 0
        ]
        parts.append(got)
        at += len(got[0])
    if not parts:
        return None
    return *_decoded(s, [np.concatenate(a) for a in zip(*parts, strict=True)], spans), spans


def _fresh_starts(s, rot, joints, spans, fps):
    """A Fresh Start block starts one frame on from the previous block's last pose and root, at their last speed, so
    nothing stalls at the seam; the difference fades out (smoothstep)."""
    fresh = blend_uids(s)
    for uid, f0, f1 in spans:
        if uid in fresh and f0 > 0:
            n = min(f1 - f0, round(FRESH_SECONDS * fps))
            x = np.arange(n) / n
            left = (1 - x * x * (3 - 2 * x))[:, None, None]
            last, back = f0 - 1, max(f0 - 2, 0)
            ahead = retarget.qmul(retarget.qmul(rot[last], retarget.qinv(rot[back])), rot[last])
            turn = retarget.qmul(ahead, retarget.qinv(rot[f0]))
            rot[f0 : f0 + n] = retarget.qmul(retarget.qpow(turn, left), rot[f0 : f0 + n])
            joints[f0 : f0 + n] += (2 * joints[last, 0] - joints[back, 0] - joints[f0, 0]) * left


def _key(obj, rot, joints, contacts, frames, times, borders, action=None, start=True, kind=None, sides=(None, None)):
    """A decoded take onto obj's action (or action): retargeted, frozen bones left alone, the legs held by IK where
    planted, unbaked or every frame as the settings say. sides: the keys' context blocks the take joins (anchor),
    the first and last of the decoded frames; then it starts and ends on the keys and is written only between them.
    Returns how far off the far seam is (anchor.join)."""
    s, t = obj.sprongmotion, blocks.settings(obj)
    if len(s.bones) != len(rig.JOINTS):  # a half-written map (an interrupted fill) is filled again
        rig.fill_map(obj)
    rig.prepare(obj)
    target = retarget.Target(obj, _s.model.offsets)
    k, off = _s.model.fps / blocks.fps(), (0.0,) * 4
    if start and any(sides):
        frames, times, first, keep, off = anchor.join(target, sides, joints, frames, times, k, _s.model.fps)
    else:
        first, keep = retarget.start_pose(obj, target, frames[0]) if start else None, np.ones(len(frames), bool)
    on = contacts[np.round(times).astype(int).clip(0, len(contacts) - 1)]
    frozen = data.frozen(obj)
    planted = on[:, [0, 2]].copy() if s.foot_lock else None  # the ankle flags: on the toes alone the ankle rolls
    for side, leg in enumerate(rig.LEGS):
        if planted is not None and frozen & {target.map.get(j) for j in leg}:
            planted[:, side] = False  # a frozen leg keeps its own keys: no IK on it
    rot, root = retarget.resample(rot, times, quat=True), retarget.resample(joints[:, 0], times)
    basis, loc = target.pose(rot, root, planted, first)
    if start and any(sides):
        anchor.ease(target, sides, frames, basis, loc, blocks.fps())
    frames, on, basis = frames[keep], on[keep], {n: q[keep] for n, q in basis.items()}
    loc, planted = loc[keep] if loc is not None else None, planted[keep] if planted is not None else None
    marks = np.r_[frames[1:][(on[1:] != on[:-1]).any(1)], [b for b in borders if frames[0] <= b <= frames[-1]]]
    tight = {}
    for side in range(2) if planted is not None else ():
        for j in rig.LEGS[side] + (0,):
            if j in target.map:
                tight[target.map[j]] = tight.get(target.map[j], False) | planted[:, side]
    basis = {name: q for name, q in basis.items() if name not in frozen}
    root_bone = target.map.get(0)
    loc = None if root_bone in frozen else loc
    if action is None and start and not t.loop and (_eases(t) or (np.diff(frames) > 1).any()):
        own = anchor.own_frames(obj, set(basis) | {root_bone})
        n = max(1, round((prefs().ease_seconds if prefs() else 0.4) * blocks.fps()))
        up = retarget.qrot(retarget.qinv(target.rest[root_bone]), target.axes[:, 1]) if loc is not None else None
        was, (frames, basis, loc, more) = frames, retarget.to_rest(frames, basis, loc, own, n, not _eases(t), up)
        marks, tight = np.r_[marks, more], {b: np.isin(frames, was[on]) for b, on in tight.items()}
    fit = (s.key_error, s.key_error, marks, s.key_frames == "SHARED", tight) if s.key_density == "UNBAKED" else None
    retarget.write(obj, frames, basis, loc, root_bone, fit, action, kind)
    return off


def action_names(s):
    """A name per block that gets its own action (enabled, with text): what block_actions pairs names with."""
    return [b.text.strip().capitalize()[:60] for b in s.blocks if b.enabled and b.text.strip()]


def block_actions(obj, names=None, asset=False):
    """Each enabled block with a take as its own action, decoded alone (loop-clean when Loop is on) from frame 1;
    the rig keeps its assigned action. With a path and root motion, each follows its own part of the curve.
    names: as action_names gives them."""
    m, s, made = model(), blocks.settings(obj), []
    named = zip([b for b in s.blocks if b.uid in _timeline_uids(s)], names or action_names(s), strict=False)
    parts = paths.parts(obj) if not s.in_place else {}
    for b, name in [(b, name) for b, name in named if _keeps(b, m)]:
        path = not b.in_place and parts.get(b.uid, False)
        with _blocks(m, obj, [(b, KEEP)], path=path, in_place=b.in_place) as (motion, first):
            data = motion.rotations(), motion.joints(), motion.contacts()
        frames = np.arange(1, b.end - b.start + 2)
        times = (frames - 1) / (frames[-1] - 1) * (first[1] - 1)
        action = bpy.data.actions.new(name)
        _key(obj, *data, frames, times, [], action, start=False, kind=keys.KEYFRAME)
        action.frame_range, action.use_frame_range, action.use_cyclic = (1, frames[-1]), True, s.loop
        action.use_fake_user = True  # unassigned until a rig plays it: kept on save
        if asset:
            action.asset_mark()
        made.append(action)
    return made


def settle(redo):
    """After an undo or redo (handler): a step that waits for its take (an edit whose take was made from the timer,
    in the next step) is passed through, so one Cmd+Z goes back to before the edit and one redo brings its take, and
    nothing is made by an undo. The step is taken from a timer: undo cannot run inside its own handler."""
    if _half():
        _s.settle = "redo" if redo else "undo"
        if not bpy.app.timers.is_registered(_settle):
            bpy.app.timers.register(_settle, first_interval=0)


def _half():
    m = _s.model
    return m and any((s := blocks.settings(o)).take_due and any(needs_take(b, m.id) for b in s.blocks) for o in rigs())


def _settle():
    step, _s.settle = _s.settle, None
    if not _half():  # a repeated Cmd+Z already left the half step: one more would undo too far
        return
    wm = bpy.context.window_manager
    window = bpy.context.temp_override(window=wm.windows[0]) if wm and wm.windows else contextlib.nullcontext()
    with window, contextlib.suppress(RuntimeError, AttributeError):  # nothing left to redo: the step stays
        getattr(bpy.ops.ed, step)()


def undo_push(message):
    wm = bpy.context.window_manager
    if bpy.app.background or not wm or not wm.windows:
        return
    with bpy.context.temp_override(window=wm.windows[0]):
        bpy.ops.ed.undo_push(message=message)
