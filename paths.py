"""A root path per rig: a Curve the runtime warps the root onto over the whole timeline. The moving blocks' lengths
fit the curve at their own gait, and Return to Start makes a way back to where the run started."""

import hashlib

import bpy
import numpy as np

from . import blocks, engine, model, quat, retarget, return_to_start, rig, takes, timeline

DEBOUNCE = 0.3  # seconds of quiet after a curve edit
SPACING = 0.1  # metres between the points handed to the runtime (at most 1024)
MOVING = 0.3  # m/s: a frame at or above it counts towards the gait (sm_motion_speed)
YAW_BUDGET = 45.0  # deg/s p95 of added yaw a walk follows without sliding feet
STRIDE = 0.1  # the path may ask +-10 % of a block's gait
REFIT = 0.05  # refit when the path length changes by more, and only change lengths off by more


class _State:
    def __init__(self):
        self.due = {}  # rig session_uid -> when to refresh
        self.own = {}  # rig session_uid -> (hash of its take's speed, its own heading per frame)


_state = _State()


def chain(verts, edges):
    """The first spline of an evaluated curve mesh: (points [n, 3] in order, closed)."""
    pairs = {tuple(sorted(e)) for e in np.asarray(edges, int).reshape(-1, 2).tolist()}
    n = 1
    while n < len(verts) and (n - 1, n) in pairs:
        n += 1
    return np.asarray(verts, float)[:n], n > 2 and (0, n - 1) in pairs


def _arc_lengths(points):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def _points_at(points, distance, n):
    """n points evenly along a polyline whose arc lengths are distance."""
    at = np.linspace(0, distance[-1], n)
    return np.stack([np.interp(at, distance, points[:, i]) for i in range(points.shape[1])], 1)


def resample(xz, closed, spacing=SPACING, most=1024):
    """Points every spacing metres along the polyline (closed: back to the first point), at most `most`."""
    points = np.vstack([xz, xz[:1]]) if closed else np.asarray(xz, float)
    distance = _arc_lengths(points)
    if distance[-1] <= 0:
        return points[:1]
    return _points_at(points, distance, max(2, min(most, int(np.ceil(distance[-1] / spacing)) + 1)))


def to_model(world, inverse, head, start, axes, ratio, src0):
    """World points [n, 3] to the model frame's (x, z) [n, 2], the inverse of how retarget.Target places the pelvis."""
    a = (np.c_[world, np.ones(len(world))] @ np.asarray(inverse).T)[:, :3] - head
    if start is not None:
        a = quat.qrot(quat.qinv(start[1]), a - start[0])
    p = a @ axes / ratio + src0
    return p[:, [0, 2]]


def length(xz):
    return float(np.linalg.norm(np.diff(xz, axis=0), axis=1).sum())


def gaits(spans, speed, fps):
    """Per block: (travel in metres, gait in m/s or 0 for a block that does not move)."""
    out = []
    for f0, f1 in spans:
        v = speed[f0:f1]
        moving = v >= MOVING
        gait = v[moving].mean() if len(v) and moving.mean() >= 0.25 else 0.0
        out.append((v.sum() / fps, gait))
    return out


def fit(spans, speed, fps, total, fixed):
    """Seconds per block so the travel adds up to the path length; None keeps a block's length."""
    found = gaits(spans, speed, fps)
    free = [gait > 0 and not f for (_, gait), f in zip(found, fixed, strict=True)]
    left = total - sum(travel for (travel, _), fr in zip(found, free, strict=True) if not fr)
    shared = sum(travel for (travel, _), fr in zip(found, free, strict=True) if fr)
    if shared <= 0:
        return [None] * len(spans)
    pairs = zip(found, free, strict=True)
    return [max(left, 0) * travel / shared / gait if fr else None for (travel, gait), fr in pairs]


def headings(root_xz, speed, fps):
    """The take's own direction of travel per frame (radians, unwrapped), held where it moves slower than 0.1 m/s."""
    sigma = 0.25 * fps  # smoothed over 0.25 s as the runtime does
    kernel = np.exp(-0.5 * (np.arange(-3 * sigma, 3 * sigma + 1) / sigma) ** 2)
    pad = np.pad(root_xz, ((len(kernel) // 2, len(kernel) // 2), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(pad[:, i], kernel / kernel.sum(), "valid") for i in range(2)], 1)
    velocity = np.gradient(smooth, axis=0)
    heading = np.arctan2(velocity[:, 0], velocity[:, 1])
    moving = speed >= 0.1
    if not moving.any():
        return np.zeros(len(heading))
    last_moving = np.maximum.accumulate(np.where(moving, np.arange(len(heading)), -1))
    return np.unwrap(heading[np.where(last_moving >= 0, last_moving, np.argmax(moving))])


def turn_rates(xz, speed, fps, own=None):
    """deg/s per frame the warp adds turning the root along the path, less the take's own turning (own: headings)."""
    distance = _arc_lengths(xz)
    heading = np.unwrap(np.arctan2(*np.diff(xz, axis=0).T))
    travel = np.r_[0, np.cumsum(speed[:-1])] / fps
    at = travel * distance[-1] / travel[-1] if travel[-1] > 0 else travel  # the root's travel stretched to the path
    theta = np.interp(at, (distance[1:] + distance[:-1]) / 2, heading) - (0 if own is None else own)
    return np.degrees(np.abs(np.gradient(theta) * fps)) if len(theta) > 1 else np.zeros(len(theta))


def notes(spans, speed, fps, xz, own=None):
    """Per block: why the path asks more than its motion follows, "" when it is within budget."""
    total = speed.sum() / fps
    stretch = length(xz) / total if total > 0 else np.inf
    rates, out = turn_rates(xz, speed, fps, own), []
    for (f0, f1), (_, gait) in zip(spans, gaits(spans, speed, fps), strict=True):
        if not gait:
            out.append("")
            continue
        moving = speed[f0:f1] >= MOVING
        why = []
        p95 = np.percentile(rates[f0:f1][moving], 95) if moving.any() else 0.0
        if p95 > YAW_BUDGET:
            why.append(f"Turn r {gait / np.radians(p95):.1f} m is too tight for this motion")
        if abs(stretch - 1) > STRIDE:
            why.append(f"Path needs {stretch:.1f}x this gait" if np.isfinite(stretch) else "Path needs a moving block")
        out.append(". ".join(why))
    return out


def path(obj):
    s = blocks.settings(obj)
    curve = s.path if s is not None else None
    return curve if curve is not None and curve.type == "CURVE" else None


def _curve_world_points(curve, depsgraph):
    """The curve's first spline in world space: (points [n, 3], closed)."""
    evaluated = curve.evaluated_get(depsgraph or bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        co = np.array([v.co for v in mesh.vertices]).reshape(-1, 3)
        points, closed = chain(co, [e.vertices for e in mesh.edges])
    finally:
        evaluated.to_mesh_clear()
    matrix = np.array(evaluated.matrix_world)
    return points @ matrix[:3, :3].T + matrix[:3, 3], closed


def curve_xz(obj, depsgraph=None):
    """The rig's path in the model frame, resampled: (points [n, 2], closed), or None."""
    curve, m = path(obj), model.loaded()
    line = blocks.timeline_blocks(blocks.settings(obj)) if curve else []
    if not line or m is None:
        return None
    rig.ensure_map(obj)
    target = retarget.Target(obj, m.offsets)
    if 0 not in target.map:
        return None
    world, closed = _curve_world_points(curve, depsgraph)
    if len(world) < 2:
        return None
    head = np.array(obj.data.bones[target.map[0]].head_local)
    start = retarget.start_pose(obj, target, line[0].start)
    xz = to_model(world, obj.matrix_world.inverted(), head, start, target.axes, target.ratio, target.src[0])
    xz = resample(xz, closed)
    return (xz, closed) if len(xz) >= 2 and length(xz) > 0.01 else None


def _path_points(xz):
    """[n, 4] (t, x, z, heading) for sm_request_set_path, times and headings free."""
    free = np.full(len(xz), np.nan)
    return np.stack([free, xz[:, 0], xz[:, 1], free], 1).astype(np.float32)


def points(obj):
    """The rig's path for sm_request_set_path, None without one."""
    got = curve_xz(obj)
    return _path_points(got[0]) if got is not None else None


def key(xz):
    return hashlib.sha1(np.round(xz, 4).tobytes()).hexdigest()[:16] if xz is not None else ""


def set_note(b, note):
    if b.path_note != note:
        b.path_note = note


def noted(obj, speed, spans):
    """After keys are written: the curve they follow, and each block's note on how well it fits."""
    s, got = blocks.settings(obj), curve_xz(obj)
    s.path_key = key(got[0] if got else None)
    fps = model.loaded().fps
    found = notes([(f0, f1) for _, f0, f1 in spans], speed, fps, got[0], _own_headings(obj, speed)) if got else []
    for (uid, *_), note in zip(spans, found, strict=False):
        i = blocks.find(s, uid)
        if i >= 0:
            set_note(s.blocks[i], "" if s.blocks[i].in_place else note)
    on_path = {uid for uid, *_ in spans}
    for b in s.blocks:
        if b.path_note and (got is None or b.uid not in on_path) and not return_to_start.walks_back(s, b):
            b.path_note = ""


def _own_headings(obj, speed):
    """The take's own travel heading, decoded without the path once per take (a curve edit keeps the take)."""
    speed_key, uid = hashlib.sha1(speed.tobytes()).hexdigest(), obj.session_uid
    if _state.own.get(uid, ("",))[0] != speed_key:
        with timeline.whole(obj, path=False) as (motion, _):
            root = motion.root()[:, [0, 2]] if motion is not None else np.zeros((len(speed), 2))
        own = headings(root, speed, model.loaded().fps) if len(root) == len(speed) else None
        _state.own[uid] = (speed_key, own)
    return _state.own[uid][1]


def blocks_key(s):
    """What a fit depends on besides the curve."""
    fitted = [(b.uid, b.end - b.start, b.text, b.seed, b.enabled, b.locked, b.back_to_start) for b in s.blocks]
    return key(np.frombuffer(repr(fitted).encode(), np.uint8))


def resize(s, b, seconds):
    """Set a block's length to seconds when it is off by more than REFIT. True if it changed."""
    (lo, hi), frames = blocks.limits(), b.end - b.start
    new = max(lo, min(hi, round(seconds * blocks.fps())))
    if abs(new - frames) <= max(1, REFIT * frames):
        return False
    i = blocks.find(s, b.uid)
    blocks.shift(s, blocks.chain(s, i), new - frames)
    b.end, b.dirty = b.start + new, True
    blocks.push(s, i)
    return True


def refit(obj):
    """Fit the unlocked moving blocks to the path when it or the blocks changed. True if some were resized."""
    s, m = blocks.settings(obj), model.load()
    if path(obj) is None:
        return return_to_start.refit(obj, s, m)
    got = curve_xz(obj) if m else None
    total = length(got[0]) if got else 0.0
    if got is None or (abs(total - s.path_length) <= REFIT * total and blocks_key(s) == s.path_fit):
        return False
    with timeline.whole(obj, path=False) as (motion, segs):
        speed = motion.speed() if motion is not None else None
    fixed = [b.locked or b.in_place for b, *_ in segs]
    seconds = fit([(f0, f1) for _, f0, f1 in segs], speed, m.fps, total, fixed) if segs else []
    changed = False
    for (b, *_), sec in zip(segs, seconds, strict=True):
        if sec is not None:
            changed |= resize(s, b, sec)
    s.path_length, s.path_fit = total, blocks_key(s)
    return changed


def refresh(obj, push):
    """Fit when needed, else key again when the curve is not what the keys follow. False when nothing was done."""
    s, m = blocks.settings(obj), model.load()
    if m is None or s is None or engine.pending(obj) or any(timeline.needs_take(b, m.id) for b in s.blocks):
        return False  # without takes yet: the job making them fits when it ends
    if refit(obj):
        engine.edited(obj)
        if push and not engine.pending(obj):
            engine.undo_push("SprongMotion path")
    elif key(got[0] if (got := curve_xz(obj)) else None) != s.path_key:
        takes.apply(obj, push=False)
    else:
        return False
    return True


def part(xz, root_xz):
    """The stretch of the path a block walks, moved to start at the origin along +Z. False when it barely moves."""
    i0 = int(np.argmin(((xz - root_xz[0]) ** 2).sum(1)))
    i1 = i0 + int(np.argmin(((xz[i0:] - root_xz[-1]) ** 2).sum(1)))
    piece = xz[i0 : i1 + 1] - xz[i0]
    if len(piece) < 2 or length(piece) < 0.1:
        return False
    a = np.arctan2(*(piece[1] - piece[0]))
    piece = piece @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    return _path_points(piece)


def parts(obj):
    """{block uid: its part of the rig's path} for per-block actions."""
    got = curve_xz(obj)
    if got is None:
        return {}
    with timeline.whole(obj) as (motion, segs):
        root = motion.root()[:, [0, 2]] if motion is not None else None
    return {b.uid: part(got[0], root[f0:f1]) for b, f0, f1 in segs} if root is not None else {}


def pin(obj, b):
    """A length set by hand wins: with a path the block is locked and made once at it."""
    if path(obj) is not None or return_to_start.walks_back(blocks.settings(obj), b):
        b["locked"] = True  # an ID write: the RNA update would refit
        b.dirty = b.remake = True


def attached(obj):
    """The rig's Path field changed (an update callback: the UI's undo step holds the result)."""
    sync()
    if path(obj) is None:
        s = blocks.settings(obj)
        s.path_key, s.path_fit, s.path_length = "", "", 0.0
        if model.load():
            takes.apply(obj, push=False)
        return
    refresh(obj, push=False)


def touch(obj):
    if path(obj) is not None:
        engine.schedule(_state.due, obj.session_uid, DEBOUNCE, _tick)


def _tick():
    for uid in engine.pop_ready(_state.due):
        obj = engine.by_uid(uid)
        if obj is not None and path(obj) is not None:
            if engine.pending(obj):
                touch(obj)
            else:
                refresh(obj, push=True)
    return 0.05 if _state.due else None


def rigs():
    """The file's rigs playing an action with a path."""
    return [o for o in engine.rigs() if path(o) is not None]


def _moved(update):
    """A curve's shape or place, or an armature's place, changed."""
    if not isinstance(update.id, bpy.types.Object):
        return False
    if update.id.type == "CURVE":
        return update.is_updated_geometry or update.is_updated_transform
    return update.id.type == "ARMATURE" and update.is_updated_transform


def _on_depsgraph(_scene, depsgraph):
    """Note which rigs' curves or rigs moved: the timer refreshes them after DEBOUNCE."""
    moved = {u.id.original.session_uid for u in depsgraph.updates if _moved(u)}
    for obj in rigs() if moved else ():
        if obj.session_uid in moved or path(obj).session_uid in moved:
            touch(obj)


def sync():
    """The depsgraph handler is there only while a rig in the file has a path."""
    engine.watch_depsgraph(_on_depsgraph, bool(rigs()))


def on_file():
    """After a file load, undo or redo: refresh every rig with a path."""
    _state.due.clear()
    sync()
    for obj in rigs():
        touch(obj)


def unregister():
    global _state
    engine.watch_depsgraph(_on_depsgraph, False)
    engine.stop_timer(_tick)
    _state = _State()


def stroke_points(gp):
    """World points [n, 3] of the last stroke on a Grease Pencil object's active layer, or None."""
    layer = gp.data.layers.active
    frame = layer.current_frame() if layer else None
    strokes = frame.drawing.strokes if frame else ()
    if not len(strokes):
        return None
    local = np.array([p.position for p in strokes[-1].points]).reshape(-1, 3)
    matrix = np.array(gp.matrix_world)
    return local @ matrix[:3, :3].T + matrix[:3, 3] if len(local) >= 2 else None


def curve_from(points, name, collection, spacing=0.5):
    """A Bezier curve object with auto handles through points [n, 3], thinned to about spacing metres apart."""
    distance = _arc_lengths(points)
    smooth = _points_at(points, distance, max(2, int(np.ceil(distance[-1] / spacing)) + 1))
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(smooth) - 1)
    spline.bezier_points.foreach_set("co", smooth.ravel())
    for p in spline.bezier_points:
        p.handle_left_type = p.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, curve)
    collection.objects.link(obj)
    return obj
