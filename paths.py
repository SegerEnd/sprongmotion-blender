"""A root path per rig: a Curve object over the whole timeline that the runtime warps the root onto
(sm_request_set_path). Attaching it or editing the curve fits the moving blocks' lengths to it at their own gait; the
lane shows where the path asks more than the motion can follow."""

import hashlib
import time

import bpy
import numpy as np

from . import blocks, engine, retarget, rig

DEBOUNCE = 0.3  # seconds of quiet after a curve edit
SPACING = 0.1  # metres between the points handed to the runtime (at most 1024)
MOVING = 0.3  # m/s: a frame at or above it counts towards the gait (sm_motion_speed)
YAW_BUDGET = 45.0  # deg/s p95 of added yaw a walk follows without sliding feet
STRIDE = 0.1  # the path may ask +-10 % of a block's gait
REFIT = 0.05  # the blocks are fitted again when the path's length changes by more, and only lengths off by more change
TURN = 2.0  # m: the least radius of a Return to Start U-turn, at the model's (human) scale like the turn budget
AHEAD = 0.5  # m: the most a Return to Start goes on straight before it turns (the body's momentum)


class _State:
    def __init__(self):
        self.due = {}  # rig session_uid -> perf_counter time to refresh at
        self.own = {}  # rig session_uid -> (key of its take's speed, its own travel heading per frame): a cache


_s = _State()


def chain(verts, edges):
    """The first spline of an evaluated curve mesh: (points [n, 3] in order, closed)."""
    pairs = {tuple(sorted(e)) for e in np.asarray(edges, int).reshape(-1, 2).tolist()}
    n = 1
    while n < len(verts) and (n - 1, n) in pairs:
        n += 1
    return np.asarray(verts, float)[:n], n > 2 and (0, n - 1) in pairs


def resample(xz, closed, spacing=SPACING, most=1024):
    """Points every spacing metres along the polyline (closed: back to the first point), at most `most`."""
    pts = np.vstack([xz, xz[:1]]) if closed else np.asarray(xz, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if s[-1] <= 0:
        return pts[:1]
    n = max(2, min(most, int(np.ceil(s[-1] / spacing)) + 1))
    at = np.linspace(0, s[-1], n)
    return np.stack([np.interp(at, s, pts[:, 0]), np.interp(at, s, pts[:, 1])], 1)


def to_model(world, inverse, head, start, axes, ratio, src0):
    """World points [n, 3] -> the model frame's (x, z) [n, 2]: the inverse of how retarget.Target places the pelvis
    (armature = matrix_world^-1 world = head + start(ratio * axes (root - rest pelvis)))."""
    a = (np.c_[world, np.ones(len(world))] @ np.asarray(inverse).T)[:, :3] - head
    if start is not None:
        a = retarget.qrot(retarget.qinv(start[1]), a - start[0])
    p = a @ axes / ratio + src0
    return p[:, [0, 2]]


def length(xz):
    return float(np.linalg.norm(np.diff(xz, axis=0), axis=1).sum())


def gaits(spans, speed, fps):
    """Per block: (travel in metres, gait in m/s or 0 for a block that does not move)."""
    out = []
    for f0, f1 in spans:
        v = speed[f0:f1]
        on = v >= MOVING
        out.append((v.sum() / fps, v[on].mean() if len(v) and on.mean() >= 0.25 else 0.0))
    return out


def fit(spans, speed, fps, total, fixed):
    """Seconds per block so the travel adds up to the path length total: the moving blocks not fixed share what the
    others leave of it as they travel now, each at its own gait. None: the block keeps its length (idle or fixed)."""
    g = gaits(spans, speed, fps)
    free = [gait > 0 and not f for (_, gait), f in zip(g, fixed, strict=True)]
    rest = total - sum(t for (t, _), fr in zip(g, free, strict=True) if not fr)
    share = sum(t for (t, _), fr in zip(g, free, strict=True) if fr)
    if share <= 0:
        return [None] * len(spans)
    return [max(rest, 0) * t / share / gait if fr else None for (t, gait), fr in zip(g, free, strict=True)]


def headings(root_xz, speed, fps):
    """The take's own direction of travel per frame (radians, unwrapped): the root smoothed over 0.25 s as the
    runtime does, held where it moves slower than 0.1 m/s."""
    sigma = 0.25 * fps
    k = np.exp(-0.5 * (np.arange(-3 * sigma, 3 * sigma + 1) / sigma) ** 2)
    pad = np.pad(root_xz, ((len(k) // 2, len(k) // 2), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(pad[:, i], k / k.sum(), "valid") for i in range(2)], 1)
    v = np.gradient(smooth, axis=0)
    h = np.arctan2(v[:, 0], v[:, 1])
    moving = speed >= 0.1
    if not moving.any():
        return np.zeros(len(h))
    last = np.maximum.accumulate(np.where(moving, np.arange(len(h)), -1))
    return np.unwrap(h[np.where(last >= 0, last, np.argmax(moving))])


def turn_rates(xz, speed, fps, own=None):
    """deg/s per frame the warp adds turning the root along the path: the path's heading change at the place the
    root's own travel (stretched to the path length) has reached, less the take's own turning (own: headings)."""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xz, axis=0), axis=1))]
    heading = np.unwrap(np.arctan2(*np.diff(xz, axis=0).T))
    travel = np.r_[0, np.cumsum(speed[:-1])] / fps
    at = travel * s[-1] / travel[-1] if travel[-1] > 0 else travel
    theta = np.interp(at, (s[1:] + s[:-1]) / 2, heading) - (0 if own is None else own)
    return np.degrees(np.abs(np.gradient(theta) * fps)) if len(theta) > 1 else np.zeros(len(theta))


def notes(spans, speed, fps, xz, own=None):
    """Per block: why the path is more than its motion follows, "" when it is within budget."""
    total = speed.sum() / fps
    stretch = length(xz) / total if total > 0 else np.inf
    rates, out = turn_rates(xz, speed, fps, own), []
    for (f0, f1), (_, gait) in zip(spans, gaits(spans, speed, fps), strict=True):
        if not gait:
            out.append("")
            continue
        on = speed[f0:f1] >= MOVING
        why = []
        p95 = np.percentile(rates[f0:f1][on], 95) if on.any() else 0.0
        if p95 > YAW_BUDGET:
            why.append(f"Turn r {gait / np.radians(p95):.1f} m is too tight for this motion")
        if abs(stretch - 1) > STRIDE:
            why.append(f"Path needs {stretch:.1f}x this gait" if np.isfinite(stretch) else "Path needs a moving block")
        out.append(". ".join(why))
    return out


def path(obj):
    s = blocks.settings(obj)
    p = s.path if s is not None else None
    return p if p is not None and p.type == "CURVE" else None


def curve_xz(obj, depsgraph=None):
    """The rig's path in the model frame, resampled: (points [n, 2], closed), or None."""
    curve, m = path(obj), engine.loaded()
    timeline = [b for b in blocks.settings(obj).blocks if b.enabled and b.text.strip()] if curve else []
    if not timeline or m is None:
        return None
    if len(obj.sprongmotion.bones) != len(rig.JOINTS):
        rig.fill_map(obj)
    target = retarget.Target(obj, m.offsets)
    if 0 not in target.map:
        return None
    ev = curve.evaluated_get(depsgraph or bpy.context.evaluated_depsgraph_get())
    mesh = ev.to_mesh()
    try:
        co = np.array([v.co for v in mesh.vertices]).reshape(-1, 3)
        pts, closed = chain(co, [e.vertices for e in mesh.edges])
    finally:
        ev.to_mesh_clear()
    if len(pts) < 2:
        return None
    world = pts @ np.array(ev.matrix_world)[:3, :3].T + np.array(ev.matrix_world)[:3, 3]
    head = np.array(obj.data.bones[target.map[0]].head_local)
    start = retarget.start_pose(obj, target, timeline[0].start)
    xz = to_model(world, obj.matrix_world.inverted(), head, start, target.axes, target.ratio, target.src[0])
    xz = resample(xz, closed)
    return (xz, closed) if len(xz) >= 2 and length(xz) > 0.01 else None


def points(obj):
    """[n, 4] (t, x, z, heading) for sm_request_set_path, times and headings free; None without a path."""
    got = curve_xz(obj)
    if got is None:
        return None
    nan = np.full(len(got[0]), np.nan)
    return np.stack([nan, got[0][:, 0], got[0][:, 1], nan], 1).astype(np.float32)


def key(xz):
    return hashlib.sha1(np.round(xz, 4).tobytes()).hexdigest()[:16] if xz is not None else ""


def noted(obj, speed, spans):
    """After keys are written (engine.apply): the curve they follow, and each block's note on how well it fits."""
    s, got = blocks.settings(obj), curve_xz(obj)
    s.path_key = key(got[0] if got else None)
    fps = engine.loaded().fps
    found = notes([(f0, f1) for _, f0, f1 in spans], speed, fps, got[0], _own(obj, speed)) if got else []
    for (uid, *_), note in zip(spans, found, strict=False):
        note = "" if (i := blocks.find(s, uid)) < 0 or s.blocks[i].in_place else note
        if i >= 0 and s.blocks[i].path_note != note:
            s.blocks[i].path_note = note
    for b in s.blocks:
        if b.path_note and (got is None or b.uid not in {u for u, *_ in spans}) and not walks_back(s, b):
            b.path_note = ""


def _own(obj, speed):
    """The take's own travel heading, decoded without the path once per take (a curve edit keeps the take)."""
    k, uid = hashlib.sha1(speed.tobytes()).hexdigest(), obj.session_uid
    if _s.own.get(uid, ("",))[0] != k:
        with engine.timeline(obj, path=False) as (motion, _):
            root = motion.root()[:, [0, 2]] if motion is not None else np.zeros((len(speed), 2))
        _s.own[uid] = (k, headings(root, speed, engine.loaded().fps) if len(root) == len(speed) else None)
    return _s.own[uid][1]


def _blocks_key(s):
    """What a fit depends on besides the curve: each block's length, text, seed, switch, lock and Return to Start."""
    return key(np.frombuffer(repr([(b.uid, b.end - b.start, b.text, b.seed, b.enabled, b.locked, b.back_to_start)
                                   for b in s.blocks]).encode(), np.uint8))  # fmt: skip


def refit(obj):
    """Fit when needed: with a path whose length moved by more than REFIT, or blocks changed since the last fit (an
    edit, a hand resize, an unlock), the unlocked moving blocks get the seconds their share of the curve takes at
    their gait. True if some were resized: they need takes (the caller's job makes them in the same undo step)."""
    s, m = blocks.settings(obj), engine.model()
    if path(obj) is None:
        return _refit_back(obj, s, m)
    got = curve_xz(obj) if m else None
    total = length(got[0]) if got else 0.0
    if got is None or (abs(total - s.path_length) <= REFIT * total and _blocks_key(s) == s.path_fit):
        return False
    with engine.timeline(obj, path=False) as (motion, segs):
        speed = motion.speed() if motion is not None else None
    fixed = [b.locked or b.in_place for b, *_ in segs]
    seconds = fit([(f0, f1) for _, f0, f1 in segs], speed, m.fps, total, fixed) if segs else []
    (lo, hi), f, changed = blocks.limits(), blocks.fps(), False
    for (b, *_), sec in zip(segs, seconds, strict=True):
        frames = b.end - b.start
        new = frames if sec is None else max(lo, min(hi, round(sec * f)))
        if abs(new - frames) > max(1, REFIT * frames):
            i = blocks.find(s, b.uid)
            blocks.shift(s, blocks.chain(s, i), new - frames)
            b.end, b.dirty, changed = b.start + new, True, True
            blocks.push(s, i)
    s.path_length, s.path_fit = total, _blocks_key(s)
    return changed


def refresh(obj, push):
    """The curve, the rig or the file changed (timer, Path field): fit when needed, else decode and key again if the
    curve is not what the keys follow. push: an undo step of its own when lengths change (a curve edit already made
    the user's). False when nothing was done."""
    s, m = blocks.settings(obj), engine.model()
    if m is None or s is None or engine.pending(obj) or any(engine.needs_take(b, m.id) for b in s.blocks):
        return False  # without takes yet: the job making them fits when it ends
    if refit(obj):
        engine.edited(obj)
        if push and not engine.pending(obj):
            engine.undo_push("SprongMotion path")
    elif key(got[0] if (got := curve_xz(obj)) else None) != s.path_key:
        engine.apply(obj, push=False)
    else:
        return False
    return True


def part(xz, root_xz):
    """The stretch of the path a block walks: the curve from where its root starts to where it ends, moved into an
    action's own frame (starting at the origin, setting off along +Z). [n, 4] points, or False when it barely moves."""
    i0 = int(np.argmin(((xz - root_xz[0]) ** 2).sum(1)))
    i1 = i0 + int(np.argmin(((xz[i0:] - root_xz[-1]) ** 2).sum(1)))
    piece = xz[i0 : i1 + 1] - xz[i0]
    if len(piece) < 2 or length(piece) < 0.1:
        return False
    a = np.arctan2(*(piece[1] - piece[0]))
    piece = piece @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    nan = np.full(len(piece), np.nan)
    return np.stack([nan, piece[:, 0], piece[:, 1], nan], 1).astype(np.float32)


def parts(obj):
    """{block uid: its part of the rig's path} for per-block actions, from the timeline as decoded with the path."""
    got = curve_xz(obj)
    if got is None:
        return {}
    with engine.timeline(obj) as (motion, segs):
        root = motion.root()[:, [0, 2]] if motion is not None else None
    return {b.uid: part(got[0], root[f0:f1]) for b, f0, f1 in segs} if root is not None else {}


def pin(obj, b):
    """A length set by hand wins: with a path, the block is locked (the fit keeps it) and remade once at it."""
    if path(obj) is not None or walks_back(blocks.settings(obj), b):
        b["locked"] = True  # an ID write: the RNA update would refit
        b.dirty = b.remake = True


def why_not_back(s, b):
    """Why a block's Return to Start does not apply, None when it does."""
    if s.path is not None:
        return "The rig's Path decides where it walks"
    return "In Place: it does not travel" if b.in_place else None


def walks_back(s, b):
    return b.back_to_start and b.enabled and b.text.strip() and why_not_back(s, b) is None


def way_back(p, h, goal, r):
    """[n, 2] points from p setting off along heading h (0 = +Z): at most AHEAD straight on (only while the goal is
    inside the turn), a turn of radius r until the goal is straight ahead, then straight to it. It turns towards the
    goal, or away from it when the goal is too close to that side."""
    u, d = np.array([np.sin(h), np.cos(h)]), goal - p
    toward = -1.0 if np.linalg.norm(d) > 1e-3 and np.sin(np.arctan2(*d) - h) < 0 else 1.0
    for side in (toward, -toward):
        w = p + r * side * np.array([u[1], -u[0]]) - goal
        need = (1.05 * r) ** 2 - w @ w + (w @ u) ** 2
        a = 0.0 if w @ w >= (1.05 * r) ** 2 else -w @ u + np.sqrt(max(need, 0))
        if a <= AHEAD:
            break
    q = p + min(a, AHEAD) * u
    c = q + r * side * np.array([u[1], -u[0]])
    psi = h + side * np.arange(0, 2 * np.pi, 0.002)
    arc = c - r * side * np.stack([np.cos(psi), -np.sin(psi)], 1)
    ahead = np.angle(np.exp(1j * (np.arctan2(*(goal - arc).T) - psi))) * side
    k = next((i for i in range(1, len(psi)) if -np.pi / 2 < ahead[i] <= 0), len(psi) - 1)
    return resample(np.vstack([p, arc[: k + 1], goal]), False, most=700)


def walk_back(obj, m, segs, first, got, fits=None):
    """A run as engine._decode decodes it (segs, their first frames, its arrays, changed in place): each block with
    Return to Start is decoded again with the root warped onto a way back to the run's first frame (way_back), timed
    to start and end on the block's own frames. The blocks before it keep their frames as decoded, it starts from them
    and ends on the start point, and the blocks after it go on from there, turned to its facing. fits: {uid: seconds
    the way back takes at the block's gait} for the fit."""
    s, rot, joints, root, _, speed = blocks.settings(obj), *got
    on = [k for k, (b, _) in enumerate(segs) if b.uid >= 0 and walks_back(s, b)]
    if not on:
        return
    start = root[first[next(k for k, (b, _) in enumerate(segs) if b.uid >= 0)], [0, 2]].copy()
    still = np.zeros(len(root), bool)  # as engine._decoded drops the travel of blocks in place
    for (b, _), f0, f1 in zip(segs, first, first[1:], strict=False):
        still[f0:f1] = b.uid >= 0 and b.in_place
    still[0] = False
    for k in on:
        b, (f0, f1), n = segs[k][0], first[k : k + 2], len(root)
        travel, gait = gaits([(f0, f1)], speed, m.fps)[0]
        if not gait:
            _note(b, "Stands still: no way back to walk")
            continue
        goal = start + (np.diff(root[:f0, [0, 2]], axis=0) * still[1:f0, None]).sum(0)
        hips, r = joints[f0, 1] - joints[f0, 2], max(TURN, gait / np.radians(YAW_BUDGET))
        back = way_back(root[f0, [0, 2]], np.arctan2(-hips[2], hips[0]), goal, r)
        lead = resample(root[: f0 + 1, [0, 2]], False, max(SPACING, length(root[: f0 + 1, [0, 2]]) / 256))[:-1]
        out = back[-1] - back[-2]
        on_to = speed[f1:].sum() / m.fps + 0.5  # past the end, so the warp does not ease the root to a stop on it
        tail = back[-1] + out / np.linalg.norm(out) * np.linspace(0.5, on_to, 1 + int(on_to)).reshape(-1, 1)
        pts = np.vstack([lead, back, tail if f1 < n else back[:0]])
        if np.linalg.norm(pts[-1] - pts[0]) < 0.01:  # the runtime closes a path whose ends meet
            pts[-1] += 0.01 * out / np.linalg.norm(out)
        t = np.full(len(pts), np.nan)
        t[len(lead)], t[len(lead) + len(back) - 1] = f0 / m.fps, (f1 - 1) / m.fps
        with engine._blocks(m, obj, segs, path=np.c_[t, pts[:, 0], pts[:, 1], t * np.nan]) as (motion, _):
            new = engine._arrays(motion)
        e0, e1 = root[f0] - new[2][f0], np.insert(goal, 1, 0) - new[2][f1 - 1]
        e = np.outer(1 - np.linspace(0, 1, f1 - f0), e0) + np.outer(np.linspace(0, 1, f1 - f0), e1)
        e[:, 1] = 0
        new[1][f0:f1] += e[:, None]
        new[2][f0:f1] += e
        if f1 < n:
            ends = [np.arctan2(*(j[f1 - 1, 1] - j[f1 - 1, 2])[[0, 2]]) for j in (new[1], joints)]
            _turn(rot[f1:], joints[f1:], root[f1:], ends[0] - ends[1], root[f1 - 1], new[2][f1 - 1])
        for a, b2 in zip((rot, joints, root), new[:3], strict=True):
            a[f0:f1] = b2[f0:f1]
        _note(b, f"Return needs {length(back) / travel:.1f}x this gait" if abs(length(back) / travel - 1) > STRIDE
              else "")  # fmt: skip
        if fits is not None:
            fits[b.uid] = length(back) / gait


def _turn(rot, joints, root, a, pivot, to):
    """Frames turned by a about pivot, then moved by to - pivot (all in place)."""
    c, s = np.cos(a), np.sin(a)
    rot[:, 0] = retarget.qmul(np.array([0, np.sin(a / 2), 0, np.cos(a / 2)]), rot[:, 0])
    for p in (joints, root):
        x, z = p[..., 0] - pivot[0], p[..., 2] - pivot[2]
        p[..., 0], p[..., 2] = c * x + s * z + to[0], -s * x + c * z + to[2]


def _note(b, note):
    if b.path_note != note:
        b.path_note = note


def _refit_back(obj, s, m):
    """refit for Return to Start: the unlocked moving blocks with it get the seconds their way back takes. Twice per
    edit at most: a take of the new length has a gait of its own (a longer walk is faster)."""
    if m is None or not any(walks_back(s, b) for b in s.blocks) or _blocks_key(s) == s.path_fit:
        return False
    again = s.path_fit != "again" + _blocks_key(s)
    fits, (lo, hi), f, changed = {}, blocks.limits(), blocks.fps(), False
    engine._decode(obj, fits)
    for uid, sec in fits.items():
        i = blocks.find(s, uid)
        b = s.blocks[i]
        frames, new = b.end - b.start, max(lo, min(hi, round(sec * f)))
        if not b.locked and abs(new - frames) > max(1, REFIT * frames):
            blocks.shift(s, blocks.chain(s, i), new - frames)
            b.end, b.dirty, changed = b.start + new, True, True
            blocks.push(s, i)
    s.path_fit = ("again" if changed and again else "") + _blocks_key(s)
    return changed


def attached(obj):
    """The rig's Path field changed (an update callback: the UI's undo step holds the result)."""
    sync()
    if path(obj) is None:
        s = blocks.settings(obj)
        s.path_key, s.path_fit, s.path_length = "", "", 0.0
        if engine.model():
            engine.apply(obj, push=False)
        return
    refresh(obj, push=False)


def touch(obj):
    if path(obj) is not None:
        _s.due[obj.session_uid] = time.perf_counter() + DEBOUNCE
        if not bpy.app.timers.is_registered(_tick):
            bpy.app.timers.register(_tick, first_interval=DEBOUNCE)


def _tick():
    now = time.perf_counter()
    for uid in [u for u, t in _s.due.items() if t <= now]:
        del _s.due[uid]
        obj = engine.by_uid(uid)
        if obj is not None and path(obj) is not None:
            if engine.pending(obj):
                touch(obj)
            else:
                refresh(obj, push=True)
    return 0.05 if _s.due else None


def rigs():
    """The file's rigs playing an action with a path."""
    return [o for o in engine.rigs() if path(o) is not None]


def _on_depsgraph(_scene, depsgraph):
    """Only notes which rigs' curves or rigs themselves moved: the refresh runs from a timer, after DEBOUNCE of
    quiet, and does nothing when the path in the rig's frame is the same (a key or playback update)."""
    moved = {u.id.original.session_uid for u in depsgraph.updates if isinstance(u.id, bpy.types.Object) and (
             u.id.type == "CURVE" and (u.is_updated_geometry or u.is_updated_transform)
             or u.id.type == "ARMATURE" and u.is_updated_transform)}  # fmt: skip
    for obj in rigs() if moved else ():
        if obj.session_uid in moved or path(obj).session_uid in moved:
            touch(obj)


def sync():
    """The depsgraph handler is there only while a rig in the file has a path."""
    handlers, want = bpy.app.handlers.depsgraph_update_post, bool(rigs())
    if want and _on_depsgraph not in handlers:
        handlers.append(_on_depsgraph)
    elif not want and _on_depsgraph in handlers:
        handlers.remove(_on_depsgraph)


def on_file():
    """After a file load, undo or redo: the handler as the file needs it; a curve that differs from what the keys were
    made for is applied again."""
    _s.due.clear()
    sync()
    for obj in rigs():
        touch(obj)


def unregister():
    global _s
    if _on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph)
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    _s = _State()


def stroke_points(gp):
    """World points [n, 3] of the last stroke on a Grease Pencil object's active layer, or None."""
    layer = gp.data.layers.active
    frame = layer.current_frame() if layer else None
    strokes = frame.drawing.strokes if frame else ()
    if not len(strokes):
        return None
    local = np.array([p.position for p in strokes[-1].points]).reshape(-1, 3)
    mw = np.array(gp.matrix_world)
    return local @ mw[:3, :3].T + mw[:3, 3] if len(local) >= 2 else None


def curve_from(points, name, collection, spacing=0.5):
    """A Bezier curve object with auto handles through points [n, 3] (thinned to about spacing metres apart)."""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    n = max(2, int(np.ceil(s[-1] / spacing)) + 1)
    at = np.linspace(0, s[-1], n)
    pts = np.stack([np.interp(at, s, points[:, i]) for i in range(3)], 1)
    data = bpy.data.curves.new(name, "CURVE")
    data.dimensions = "3D"
    spline = data.splines.new("BEZIER")
    spline.bezier_points.add(len(pts) - 1)
    spline.bezier_points.foreach_set("co", pts.ravel())
    for p in spline.bezier_points:
        p.handle_left_type = p.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    return obj
