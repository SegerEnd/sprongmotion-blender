import os
import tempfile

import bpy
import numpy as np
from bl_ext.smtest.sprongmotion import blocks, generate, model, takes, timeline_motion
from test_generate import all_codes, codes, curve, setup


def test_variants_bvh_bake():
    obj = setup()
    obj.sprongmotion.variants = 3
    generate.run_now(obj)
    b = blocks.settings(obj).blocks[0]
    assert b.variants == 3 and not (codes(b, 0) == codes(b, 1)).all()
    later, keys = all_codes(obj)[1], [k.co.y for k in curve(obj).keyframe_points]
    b.variant = 2
    generate.run_now(obj)
    assert not blocks.settings(obj).blocks[1].dirty and (all_codes(obj)[1] == later).all(), "the next block stays"
    assert keys != [k.co.y for k in curve(obj).keyframe_points], "the other take is on the armature"
    path = os.path.join(tempfile.mkdtemp(), "take.bvh")
    assert bpy.ops.sm.export_bvh(filepath=path) == {"FINISHED"}
    with open(path) as f:
        text = f.read()
    assert text.startswith("HIERARCHY") and f"Frames: {2 * sum(x.tokens for x in blocks.settings(obj).blocks)}" in text
    n = len(curve(obj).keyframe_points)
    assert bpy.ops.sm.bake(keys="4") == {"FINISHED"}
    keys = curve(obj).keyframe_points
    assert not len(blocks.settings(obj).blocks) and {k.type for k in keys} == {"KEYFRAME"} and len(keys) < n


def test_foot_lock_and_in_place_come_from_the_runtime():
    obj = setup()
    generate.run_now(obj)
    locked = [k.co.y for k in curve(obj).keyframe_points]
    obj.sprongmotion.foot_lock = False
    generate.run_now(obj)
    assert locked != [k.co.y for k in curve(obj).keyframe_points]
    blocks.settings(obj).in_place = True
    generate.run_now(obj)
    heads = []
    for f in range(1, blocks.settings(obj).blocks[-1].end, 10):
        bpy.context.scene.frame_set(f)
        heads.append(np.array(obj.pose.bones["pelvis"].head)[:2])
    assert np.ptp(heads, axis=0).max() < 0.01


def ankle_track(obj, side=0):
    """Scene frames, the world ankle head per frame and the runtime's ankle contact per frame."""
    with timeline_motion.whole(obj) as (motion, segs):
        contacts = motion.contacts()
    frames, times = takes.timing(segs)
    planted = contacts[np.round(times).astype(int).clip(0, len(contacts) - 1), 2 * side]
    pos = []
    for f in frames:
        bpy.context.scene.frame_set(int(f))
        pos.append(np.array(obj.matrix_world @ obj.pose.bones[("l_ankle", "r_ankle")[side]].head))
    return np.array(pos), planted


def slide(pos, planted):
    both = planted[1:] & planted[:-1]
    return np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)[both].mean() * 100


def test_planted_feet_hold_still_with_foot_lock():
    obj = setup()
    for k, b in enumerate(blocks.settings(obj).blocks):
        b.seed = 11 + k
    obj.sprongmotion.foot_lock = False
    generate.run_now(obj)
    loose = [slide(*ankle_track(obj, side)) for side in (0, 1)]
    obj.sprongmotion.foot_lock = True
    generate.run_now(obj)
    for side in (0, 1):
        pos, planted = ankle_track(obj, side)
        assert slide(pos, planted) < 0.05 < loose[side], (slide(pos, planted), loose[side])
        hip = np.array([obj.matrix_world @ obj.pose.bones[("l_hip", "r_hip")[side]].head])
        assert np.isfinite(pos).all() and np.isfinite(hip).all()


def test_the_take_starts_where_the_rig_was_keyed():
    obj = setup()
    generate.run_now(obj)
    pelvis = obj.pose.bones["pelvis"]
    end = blocks.settings(obj).blocks[-1].end

    def path():
        heads = []
        for f in (1, end):
            bpy.context.scene.frame_set(f)
            heads.append(np.array(obj.matrix_world @ pelvis.head)[:2])
        return heads

    a0, a1 = path()
    pelvis.location = (0.4, 0.0, 0.3)
    pelvis.rotation_mode = "QUATERNION"
    pelvis.rotation_quaternion = (np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0)
    pelvis.keyframe_insert("location", frame=0)
    pelvis.keyframe_insert("rotation_quaternion", frame=0)
    bpy.context.scene.frame_set(0)
    keyed = np.array(obj.matrix_world @ pelvis.head)[:2]
    generate.run_now(obj)
    b0, b1 = path()
    assert np.linalg.norm(b0 - keyed) < np.linalg.norm(a0 - keyed) and np.linalg.norm(b0 - keyed) < 0.05
    da, db = a1 - a0, b1 - b0
    turn = np.degrees(np.arctan2(da[0] * db[1] - da[1] * db[0], da @ db))
    assert abs(abs(turn) - 90) < 5, turn


def lowest(obj, b):
    z = []
    for f in range(b.start, b.end):
        bpy.context.scene.frame_set(f)
        z.append(min((obj.matrix_world @ pb.head).z for pb in obj.pose.bones))
    return np.array(z)


def test_floor_keeps_a_somersault_above_the_floor():
    assert bpy.ops.sm.add_rig() == {"FINISHED"}, model.error()
    obj = bpy.context.active_object
    obj.sprongmotion.key_density = "EXACT"
    b = blocks.add(obj, "does a somersault", 3)
    b.seed = 2  # a random seed now and then makes a somersault that stays above the floor
    obj.sprongmotion.floor_clamp = False
    generate.run_now(obj)
    assert lowest(obj, b).min() < -0.05, "the test needs a take that goes below the floor"
    obj.sprongmotion.floor_clamp = True
    generate.run_now(obj)
    assert not model.error(), model.error()
    assert lowest(obj, b).min() > -0.005
