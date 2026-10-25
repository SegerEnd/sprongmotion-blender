import bpy
import numpy as np
from bl_ext.smtest.sprongmotion import blocks, generate, model, retarget, rig_data, takes
from mathutils import Vector


def prefs():
    return bpy.context.preferences.addons["bl_ext.smtest.sprongmotion"].preferences


def setup(gap=0, after="HOLD"):
    prefs().after_block, prefs().ease_seconds = after, 0.4
    bpy.context.scene.render.fps = 24
    bpy.context.scene.frame_current = 1
    bpy.ops.sm.add_rig()
    obj = bpy.context.active_object
    obj.sprongmotion.auto = False
    blocks.add(obj, "a person walks forward", 2).seed = 2
    if gap:
        b = blocks.add(obj, "a person waves", 2)
        b.seed, b.start, b.end = 3, b.start + gap, b.end + gap
    generate.run_now(obj)
    return obj


def curves(obj):
    return [fc for fc in retarget.channelbag(obj).fcurves if fc.data_path.startswith("pose.bones[")]


def keyset(obj):
    return {(fc.data_path, fc.array_index): [(round(k.co.x, 4), round(k.co.y, 6)) for k in fc.keyframe_points]
            for fc in curves(obj)}  # fmt: skip


def pose(obj, f):
    """[curves] values at frame f, a quaternion's w as |w| (either sign is the same rotation)."""
    return np.array([abs(fc.evaluate(f)) if fc.data_path.endswith("quaternion") and fc.array_index == 0
                     else fc.evaluate(f) for fc in curves(obj)])  # fmt: skip


def rest(obj, at):
    """[curves] at rest after frame at: unit rotations, the root keeping its travel over the ground."""
    cs = curves(obj)
    out = np.array([float(fc.data_path.endswith("quaternion") and fc.array_index == 0) for fc in cs])
    for name in {fc.data_path.split('"')[1] for fc in cs if fc.data_path.endswith(".location")}:
        up = np.array((obj.data.bones[name].matrix_local.to_3x3().inverted() @ Vector((0, 0, 1))).normalized())
        idx = {fc.array_index: i for i, fc in enumerate(cs) if fc.data_path == f'pose.bones["{name}"].location'}
        held = np.array([cs[idx[k]].evaluate(at) if k in idx else 0.0 for k in range(3)])
        for k, i in idx.items():
            out[i] = (held - (held @ up) * up)[k]
    return out


def steps(obj, lo, hi):
    """The largest change of any curve from one frame to the next over lo..hi."""
    return np.abs(np.diff([pose(obj, f) for f in range(lo, hi + 1)], axis=0)).max()


def ease_frames():
    return round(prefs().ease_seconds * blocks.fps())


def test_hold_keeps_the_keys():
    obj = setup()
    before = keyset(obj)
    prefs().after_block = "REST"
    assert keyset(obj) != before
    prefs().after_block = "HOLD"
    assert keyset(obj) == before


def test_ease_to_rest_after_the_last_block():
    obj = setup()
    end, held = blocks.settings(obj).blocks[-1].end, pose(setup(), blocks.settings(obj).blocks[-1].end)
    prefs().after_block = "REST"
    n = ease_frames()
    assert {k.type for fc in curves(obj) for k in fc.keyframe_points} == {"GENERATED"}
    for f in (end + n, end + n + 30):
        assert np.abs(pose(obj, f) - rest(obj, end)).max() < 1e-4, f
    assert np.abs(pose(obj, end) - held).max() < 1e-3, "the block's own pose at its end"
    assert steps(obj, end, end + n) < 2.0 * steps(obj, end - n, end) + 0.02, "no step at the block's end"
    prefs().ease_seconds = 1.0
    assert np.abs(pose(obj, end + ease_frames()) - rest(obj, end)).max() < 1e-4
    assert np.abs(pose(obj, end + n) - rest(obj, end)).max() > 1e-3, "a longer ease"


def test_gaps_between_blocks():
    n = round(0.4 * 24)
    for gap in (3 * n, n):
        obj = setup(gap=gap, after="REST")
        a, b = blocks.settings(obj).blocks
        assert b.start - a.end == gap
        if gap > 2 * n:
            assert np.abs(pose(obj, a.end + gap // 2) - rest(obj, a.end)).max() < 1e-4, "rest in a long gap"
        far = max(np.abs(pose(obj, f) - rest(obj, a.end)).max() for f in (a.end, b.start))
        smooth = 1.5 * far / min(n, gap / 2)  # the steepest a smoothstep over the ease gets
        assert steps(obj, a.end - 1, b.start + 1) < smooth + steps(obj, a.end - n, a.end - 1), f"a snap in {gap}"


def test_loop_has_no_ease():
    obj = setup()
    blocks.settings(obj).loop = True
    generate.run_now(obj)
    before = keyset(obj)
    prefs().after_block = "REST"
    assert keyset(obj) == before


def test_frozen_bones_and_own_keys_stay():
    obj = setup()
    rig_data.freeze(obj, ["l_elbow"])
    elbow = {k: v for k, v in keyset(obj).items() if "l_elbow" in k[0]}
    prefs().after_block = "REST"
    assert {k: v for k, v in keyset(obj).items() if "l_elbow" in k[0]} == elbow
    end, n = blocks.settings(obj).blocks[-1].end, ease_frames()
    knee = next(fc for fc in curves(obj) if fc.data_path == 'pose.bones["l_knee"].rotation_quaternion')
    k = knee.keyframe_points.insert(end + 4 * n, 0.9)
    k.type = "KEYFRAME"
    takes.apply(obj, push=False)
    assert (end + 4 * n, 0.9) in [(k.co.x, round(k.co.y, 6)) for k in knee.keyframe_points], "the own key stays"
    assert max(k.co.x for fc in curves(obj) if "l_knee" not in fc.data_path for k in fc.keyframe_points) <= end, (
        "no ease where own keys follow"
    )


def test_preference_change_makes_no_take():
    obj = setup()
    m = model.load()
    codes = [blocks.codes(b, m.id).tolist() for b in blocks.settings(obj).blocks]
    made, pushes, take, push = [], [], generate._Job.take, generate.undo_push
    generate._Job.take = lambda *a: made.append(a)
    generate.undo_push = pushes.append
    try:
        prefs().after_block = "REST"
    finally:
        generate._Job.take, generate.undo_push = take, push
    assert not made and pushes == ["SprongMotion: After a Block"]
    assert [blocks.codes(b, m.id).tolist() for b in blocks.settings(obj).blocks] == codes
