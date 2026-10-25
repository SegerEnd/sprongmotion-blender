import time

import addon_utils
import bpy
import numpy as np
from bl_ext.smtest.sprongmotion import blocks, engine, model, paths, timeline
from test_cleanup import PKG, add_on_props

WALK = "a person walks forward"
_push = engine.undo_push


def rig(*texts):
    bpy.context.scene.render.fps = 24
    bpy.context.scene.frame_current = 1
    assert bpy.ops.sm.add_rig() == {"FINISHED"}, model.error()
    obj = bpy.context.active_object
    s = obj.sprongmotion
    s.auto, s.key_density = False, "EXACT"
    for text, seconds in texts:
        blocks.add(obj, text, seconds).seed = 5
    engine.run_now(obj)
    s.auto = True
    return obj


def s_curve(name="Path", scale=1.0):
    t = np.linspace(0, 1, 40)
    pts = np.c_[0.4 * np.sin(t * 3), 2.5 * t, np.zeros_like(t)] * scale
    return paths.curve_from(pts, name, bpy.context.scene.collection, spacing=0.25)


def lengths(obj):
    return [b.end - b.start for b in blocks.settings(obj).blocks]


def moving(obj):
    """Per block: whether its own motion travels, as the fit sees it (only those get a new length)."""
    *_, speed, spans = timeline.decode(obj)
    return [gait > 0 for _, gait in paths.gaits([(f0, f1) for _, f0, f1 in spans], speed, model.load().fps)]


def watch():
    """Counts what the model and undo see: (blocks made per model call, undo pushes)."""
    m, made, out = model.load(), [], []
    run = m.blocks

    def counted(texts, seconds, seeds, keep, *a):
        made.append(sum(k is None for k in keep))
        return run(texts, seconds, seeds, keep, *a)

    m.blocks, engine.undo_push = counted, out.append
    return made, out


def unwatch():
    vars(model.load()).pop("blocks", None)
    engine.undo_push = _push


def pelvis_xy(obj, frames):
    scene, out = bpy.context.scene, []
    for f in frames:
        scene.frame_set(int(f))
        out.append((obj.matrix_world @ obj.pose.bones["pelvis"].head).xy[:])
    return np.array(out)


def off_curve(xy, curve):
    ev = curve.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = ev.to_mesh()
    c = np.array([(ev.matrix_world @ v.co).xy[:] for v in me.vertices])
    ev.to_mesh_clear()
    a, b = c[:-1], c[1:]
    t = np.clip(np.einsum("pnk,nk->pn", xy[:, None] - a, b - a) / ((b - a) ** 2).sum(1), 0, 1)
    return np.linalg.norm(a + t[..., None] * (b - a) - xy[:, None], axis=2).min(1)


def smooth(xy, sigma=6):
    """The pelvis without its stride sway: Gaussian over 0.25 s at 24 fps, as the runtime smooths the root."""
    k = np.exp(-0.5 * (np.arange(-3 * sigma, 3 * sigma + 1) / sigma) ** 2)
    pad = np.pad(xy, ((3 * sigma, 3 * sigma), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, i], k / k.sum(), "valid") for i in range(2)], 1)


def decode_only(obj, change):
    """change(), then the debounced refresh as the timer runs it: (blocks made per model call, undo pushes, ms)."""
    change()
    bpy.context.view_layer.update()
    assert obj.session_uid in paths._state.due, "the depsgraph handler saw it move"
    paths._state.due[obj.session_uid] = 0
    made, out = watch()
    try:
        t0 = time.perf_counter()
        paths._tick()
        return made, out, 1000 * (time.perf_counter() - t0)
    finally:
        unwatch()


def test_attach_fits_the_walk_the_root_follows_the_curve_and_moves_only_decode():
    """A straight path away from the rig and at an angle: the smoothed pelvis (no stride sway) lies on it, from its
    start. Bends are the runtime's own tests. Moving the curve or the rig keeps the lengths and only decodes."""
    obj = rig((WALK, 2), ("a person waves", 1.5))
    s, before, on = blocks.settings(obj), lengths(obj), moving(obj)
    t = np.linspace(0, 1, 9)[:, None]
    curve = paths.curve_from([1.0, 0.5, 0] + t * [1.5, 2.6, 0], "Line", bpy.context.scene.collection)
    made, out = watch()
    try:
        s.path = curve
    finally:
        unwatch()
    assert made[:2] == [0, sum(on)] and sum(made) == sum(on) and not out, "attaching and its fit: one undo step"
    after = lengths(obj)
    assert after[0] != before[0] and (after[1] == before[1]) != on[1], (before, after, on)
    assert s.blocks[1].start == s.blocks[0].end and s.path_key
    frames = range(s.blocks[0].start, s.blocks[-1].end + 1)
    xy = pelvis_xy(obj, frames)
    d = off_curve(smooth(xy)[18:-18], curve)
    print(f"> root off the curve: max {100 * d.max():.1f} cm, {100 * off_curve(xy, curve).max():.1f} with the sway")
    assert d.max() < 0.02 and np.linalg.norm(xy[0] - [1, 0.5]) < 0.02, (d.max(), xy[0])

    made, out, ms = decode_only(obj, lambda: setattr(curve.location, "x", curve.location.x + 1))
    print(f"> curve edit to keys {ms:.0f} ms (decode only, {model.THREADS or 'all'} model threads)")
    assert made and not any(made) and not out and lengths(obj) == after, (made, out)
    assert np.allclose(pelvis_xy(obj, frames[-1:])[0], xy[-1] + [1, 0], atol=0.02)

    def move_and_rename():
        obj.location.y -= 2
        bpy.context.view_layer.update()
        obj.name = "Renamed Rig"

    made, out, _ = decode_only(obj, move_and_rename)
    assert made and not any(made) and not out and lengths(obj) == after, (made, out)
    assert np.allclose(pelvis_xy(obj, frames[-1:])[0], xy[-1] + [1, 0], atol=0.02), "the root stays on the curve"


def test_a_longer_curve_refits_in_one_undo_step_and_so_does_a_text_edit():
    obj = rig((WALK, 2), ("a person waves", 1.5), (WALK, 2))
    s = blocks.settings(obj)
    s.path = s_curve()
    s.blocks[2].locked = True
    before, on = lengths(obj), moving(obj)
    s.path.scale = (1.6, 1.6, 1.6)
    made, out = watch()
    try:
        assert paths.refresh(obj, push=True)
    finally:
        unwatch()
    after = lengths(obj)
    assert after[0] > before[0] and after[2] == before[2] and (after[1] == before[1]) != on[1], (before, after, on)
    assert made[:2] == [0, 1 + on[1]], made
    assert out == ["SprongMotion path"], out
    assert "Path needs" in s.blocks[2].path_note, [b.path_note for b in s.blocks]
    made, out = watch()
    try:
        s.blocks[0].text = "a person walks forward slowly"
    finally:
        unwatch()
    print(f"> text edit with a path: blocks made per call {made}, lengths {after} -> {lengths(obj)}")
    assert made[:2] == [1, 0] and not out, "the take, then the fit's decode, inside the edit's own undo step"
    resized = [n != m for n, m in zip(lengths(obj), after, strict=True)]
    assert sum(made) == 1 + sum(resized) and not resized[2] and (resized[1] <= on[1]), (resized, made)


def test_a_tight_turn_goes_amber():
    obj = rig((WALK, 3))
    a = np.linspace(0, 1.5 * np.pi, 40)
    blocks.settings(obj).path = paths.curve_from(
        np.c_[np.cos(a) - 1, np.sin(a), 0 * a] * 0.8, "Tight", bpy.context.scene.collection, 0.2
    )
    assert "too tight" in blocks.settings(obj).blocks[0].path_note, blocks.settings(obj).blocks[0].path_note


def test_path_from_stroke_then_remove_and_unregister_leave_no_handler():
    obj = rig((WALK, 1.5))
    bpy.ops.object.grease_pencil_add(type="STROKE")
    gp = bpy.context.active_object
    bpy.context.view_layer.objects.active = obj
    gp.select_set(True)
    assert bpy.ops.sm.path_from_stroke() == {"FINISHED"}
    curve = blocks.settings(obj).path
    assert curve and curve.type == "CURVE" and len(curve.data.splines[0].bezier_points) >= 2
    assert blocks.settings(obj).path_key and paths._on_depsgraph in bpy.app.handlers.depsgraph_update_post
    assert bpy.ops.sm.remove_data() == {"FINISHED"}
    assert paths._on_depsgraph not in bpy.app.handlers.depsgraph_update_post and not add_on_props()
    blocks.settings(obj, create=True).path = curve
    assert paths._on_depsgraph in bpy.app.handlers.depsgraph_update_post
    prefs = bpy.context.preferences.addons[PKG].preferences
    saved = prefs.library_path, prefs.model_path
    addon_utils.disable(PKG)
    left = [h for n in dir(bpy.app.handlers) if isinstance(hs := getattr(bpy.app.handlers, n), list) for h in hs]
    assert not [h for h in left if "sprongmotion" in getattr(h, "__module__", "")], left
    assert not bpy.app.timers.is_registered(paths._tick)
    addon_utils.enable(PKG, default_set=True)
    prefs = bpy.context.preferences.addons[PKG].preferences
    prefs.library_path, prefs.model_path = saved
