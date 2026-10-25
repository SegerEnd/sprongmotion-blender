"""Making takes. Small edits generate at once, so the edit and its take are one undo step. Bigger jobs run from a
timer, one block per tick, so the UI stays live and Cancel works between blocks.
"""

import contextlib
import time
import traceback
from collections import namedtuple

import bpy
import numpy as np

from . import blocks, continue_keys, model, paths, return_to_start, rig, rig_data, takes, timeline_motion

DEBOUNCE = 0.25  # seconds of quiet after an edit before a timed regenerate
SYNC_BLOCKS = 3  # single-take blocks generated inside the edit itself (about 0.1 s each)

Progress = namedtuple("Progress", "rig uid done total")  # rig: the armature's session_uid, uid: the block being made


class _State:
    def __init__(self):
        self.job = None
        self.due = {}  # rig session_uid -> when its timed regenerate starts
        self.edits = {}  # rig session_uid -> edit count
        self.hold, self.held = 0, set()
        self.settle = None  # "undo" or "redo": the step settle takes next


_state = _State()


def by_uid(uid):
    """The object with this session_uid (kept across undo and renames), or None."""
    return next((o for o in bpy.data.objects if o.session_uid == uid), None)


def schedule(due, uid, delay, tick):
    """Run the timer tick for this rig after delay seconds of quiet."""
    due[uid] = time.perf_counter() + delay
    if not bpy.app.timers.is_registered(tick):
        bpy.app.timers.register(tick, first_interval=delay)


def pop_ready(due):
    """The rigs whose time in due has come, taken out of it."""
    now = time.perf_counter()
    ready = [uid for uid, at in due.items() if at <= now]
    for uid in ready:
        del due[uid]
    return ready


def watch_depsgraph(handler, on):
    """Add or remove a depsgraph handler."""
    handlers = bpy.app.handlers.depsgraph_update_post
    if on and handler not in handlers:
        handlers.append(handler)
    elif not on and handler in handlers:
        handlers.remove(handler)


def stop_timer(tick):
    if bpy.app.timers.is_registered(tick):
        bpy.app.timers.unregister(tick)


def rigs():
    """The file's local armatures playing an action with blocks."""
    return [o for o in bpy.data.objects if o.library is None and blocks.settings(o) is not None]


def reapply():
    """Key every rig again from its takes as one undo step (a preference changed)."""
    if model.load() is None:
        return
    found = [o for o in rigs() if len(blocks.settings(o).blocks) and not pending(o)]
    for obj in found:
        takes.apply(obj, push=False)
    if found:
        undo_push("SprongMotion: After a Block")


def shutdown():
    model.unload()
    stop_timer(_tick)
    stop_timer(_settle)


@contextlib.contextmanager
def hold():
    """Edits inside count as one: a single regenerate when the outermost hold ends."""
    _state.hold += 1
    try:
        yield
    finally:
        _state.hold -= 1
    if not _state.hold:  # not reached when the edits raised: nothing regenerates then
        uids, _state.held = _state.held, set()
        for obj in filter(None, map(by_uid, uids)):
            edited(obj)


def edited(obj, wait=False, force=False):
    """obj's blocks changed. Small jobs run now, big ones and slider drags (wait) from the timer."""
    uid = obj.session_uid
    _state.edits[uid] = _state.edits.get(uid, 0) + 1
    if _state.hold:
        _state.held.add(uid)
        return
    r, s, m = obj.sprongmotion, blocks.settings(obj), model.load()
    if not (force or r.auto) or m is None or s is None:
        return
    if busy(obj):
        cancel()
    small = r.variants == 1 and sum(timeline_motion.needs_take(b, m.id) for b in s.blocks) <= SYNC_BLOCKS
    if small and not wait and not busy():
        return run_now(obj)
    s.take_due = True  # the edit's undo step: its take comes in a later one (settle)
    schedule(_state.due, uid, DEBOUNCE, _tick)


def run_now(obj):
    cancel()
    job = _Job(obj.session_uid, push=False, per_block=False)
    while not _advance(job):
        pass
    blocks.redraw()


def busy(obj=None):
    return _state.job is not None and (obj is None or _state.job.rig == obj.session_uid)


def pending(obj):
    """A take for obj is being made or waits for the timer."""
    return busy(obj) or obj.session_uid in _state.due


def progress():
    job = _state.job
    if job is None or job.step >= len(job.units):
        return None
    return Progress(job.rig, job.units[job.step][0][0], job.step, len(job.units))


def cancel():
    if _state.job:
        _state.job.stop()
    _state.job = None
    _state.due.clear()


def _advance(job):
    """Advance a job. A failure ends it with one message, never a timer that raises every tick."""
    try:
        return job.advance()
    except Exception as e:
        traceback.print_exc()
        job.stop()
        model.set_error(f"SprongMotion: {e}")
        return True


def _tick():
    now = time.perf_counter()
    if _state.job is None:
        ready = next((uid for uid, at in _state.due.items() if at <= now), None)
        if ready:
            del _state.due[ready]
            _state.job = _Job(ready, push=True)
    if _state.job is not None:
        if _advance(_state.job):
            _state.job = None
        blocks.redraw()
    return 0.01 if _state.job or _state.due else None


class _Job:
    """Makes the takes of every block that needs one, one generate call per advance; an edit starts it over."""

    def __init__(self, uid, push, per_block=True):
        self.rig, self.push, self.per_block = uid, push, per_block
        self.action = _action_uid(by_uid(uid))
        self.edits = object()  # never equals an edit count, so the first advance plans
        self.stop()

    def stop(self):
        self.todo, self.units, self.takes, self.step, self.decoded = [], [], {}, 0, None

    def advance(self):
        """Make one unit, or at the end store the takes and key the rig. True when the job is over."""
        obj, m = by_uid(self.rig), model.load()
        if obj is not None and (why := rig_data.why_not_editable(obj)):
            raise RuntimeError(why)
        s = blocks.settings(obj) if obj is not None else None
        if s is None or m is None or _action_uid(obj) != self.action:
            return True  # after an action switch the takes made so far are for the other action's blocks
        if self.edits != _state.edits.get(self.rig):
            self.plan(obj, m)
        if self.step < len(self.units):
            self.take(obj, m, *self.units[self.step])
            self.step += 1
            return False
        for uid, made in self.takes.items():
            if (i := blocks.find(s, uid)) >= 0:
                blocks.store(s.blocks[i], np.stack(made), m.id)
                s.blocks[i].anchor_key = self.joins.get(uid, "|")
        if paths.refit(obj):
            self.edits = object()  # the resized blocks are made in this same job, in the edit's undo step
            return False
        for b in s.blocks:
            model.unknown_words(b.text)
        if s.take_due:
            s.take_due = False
        takes.apply(obj, self.push, self.decoded)
        _apply_others(obj)
        model.set_error("")
        return True

    def plan(self, obj, m):
        """The units to make: (block uids in one call, take number). Blocks whose keys around them changed need one."""
        s = blocks.settings(obj)
        self.edits, self.takes, self.step, self.decoded = _state.edits.get(self.rig), {}, 0, None
        rig.ensure_map(obj)
        self.sides = continue_keys.sides(obj, m)
        self.joins = continue_keys.block_keys(s, self.sides)
        for b in s.blocks:
            if b.anchor_key != self.joins.get(b.uid, "|"):
                b.dirty = True
        self.todo = [b.uid for b in s.blocks if timeline_motion.needs_take(b, m.id)]
        fresh = timeline_motion.fresh_uids(s)
        run_of = {uid: k for k, run in enumerate(timeline_motion.run_uids(s)) for uid in run}
        first_takes = []
        for uid in self.todo:
            if self.per_block or not first_takes or uid in fresh or run_of[uid] != run_of[first_takes[-1][0][-1]]:
                first_takes.append(((uid,), 0))  # a fresh block starts a call: no block before it in it
            else:
                first_takes[-1] = ((*first_takes[-1][0], uid), 0)
        more_takes = [((uid,), v) for v in range(1, obj.sprongmotion.variants) for uid in self.todo]
        self.units = first_takes + more_takes

    def take(self, obj, m, make, variant):
        """Make the blocks in make; a later block goes in as a short stand-in, so the edges match one whole call."""
        s = blocks.settings(obj)
        firsts, line = {uid: t[0] for uid, t in self.takes.items()}, timeline_motion.uids(s)
        later = {b.uid for b in s.blocks if b.uid in self.todo and b.uid not in firsts and b.uid not in make}
        run = next(r for r in timeline_motion.run_uids(s) if make[0] in r)
        segs = []
        for b in s.blocks:
            if b.uid not in run:
                continue
            if b.uid in make or b.uid in firsts or (b.uid not in self.todo and timeline_motion.has_take(b, m)):
                segs.append((b, timeline_motion.MAKE if b.uid in make else timeline_motion.KEEP))
            elif b.uid in later and ((segs and segs[-1][1] == timeline_motion.MAKE) or (s.loop and b.uid == line[-1])):
                segs.append((b, timeline_motion.STUB))
        if make[0] in timeline_motion.fresh_uids(s):
            segs = segs[[b.uid for b, _ in segs].index(make[0]) :]
        whole = obj.sprongmotion.variants == 1 and [b.uid for b, _ in segs] == line
        whole = whole and not any(return_to_start.walks_back(s, b) for b in s.blocks)
        segs = timeline_motion.with_sides(segs, self.sides, line)
        with timeline_motion.run_blocks(m, obj, segs, variant, firsts) as (motion, first):
            codes = motion.codes()
            if whole:
                spans = timeline_motion.block_spans(segs, first)
                self.decoded = (*timeline_motion.without_travel(s, timeline_motion.arrays(motion), spans), spans)
            else:
                self.decoded = None
        for (b, kind), f0, f1 in zip(segs, first, first[1:], strict=False):
            if kind == timeline_motion.MAKE:
                self.takes.setdefault(b.uid, []).append(codes[f0 // m.fpt : f1 // m.fpt])


def _action_uid(obj):
    ad = obj.animation_data if obj else None
    return ad.action.session_uid if ad and ad.action else None


def _apply_others(obj):
    """Key every other rig playing the same action with its own bone map."""
    for other in blocks.players(obj.animation_data.action):
        if other != obj and not pending(other):
            takes.apply(other, push=False)


def settle(redo):
    """After an undo or redo, step past a state that waits for its take, so one Cmd+Z undoes the edit and its take."""
    if _waits_for_take():
        _state.settle = "redo" if redo else "undo"
        if not bpy.app.timers.is_registered(_settle):
            bpy.app.timers.register(_settle, first_interval=0)  # undo cannot run inside its own handler


def _waits_for_take():
    m = model.loaded()
    if not m:
        return m
    for obj in rigs():
        s = blocks.settings(obj)
        if s.take_due and any(timeline_motion.needs_take(b, m.id) for b in s.blocks):
            return True
    return False


def _settle():
    step, _state.settle = _state.settle, None
    if not _waits_for_take():  # a repeated Cmd+Z already left the half step: one more would undo too far
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
