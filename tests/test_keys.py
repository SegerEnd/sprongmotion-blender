import bpy
import numpy as np
from bl_ext.smtest.sprongmotion import keys, retarget


def rig_with_bone():
    arm = bpy.data.armatures.new("K")
    obj = bpy.data.objects.new("K", arm)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    arm.edit_bones.new("b").tail = (0, 0, 1)
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def listed(obj, index=0):
    fc = retarget.channelbag(obj).fcurves.find('pose.bones["b"].rotation_quaternion', index=index)
    return [(k.co.x, k.co.y, k.type, k.interpolation) for k in fc.keyframe_points]


def test_generated_keys_replace_their_range():
    obj = rig_with_bone()
    q = np.tile([0, 0, 0, 1.0], (11, 1))
    retarget.write(obj, np.arange(10, 21), {"b": q})
    got = listed(obj, 0)
    assert len(got) == 11 and {k[2] for k in got} == {"GENERATED"} and {k[3] for k in got} == {"LINEAR"}
    fc = retarget.channelbag(obj).fcurves.find('pose.bones["b"].rotation_quaternion', index=0)
    fc.keyframe_points.insert(5, 0.5).type = "KEYFRAME"
    fc.keyframe_points.insert(15, 0.7).type = "KEYFRAME"
    retarget.write(obj, np.arange(12, 18), {"b": q[:6] * -1})
    got = listed(obj, 0)
    assert [k[0] for k in got] == [5, 12, 13, 14, 15, 16, 17], "user key outside stays, old take goes"
    assert got[0][2] == "KEYFRAME" and got[1][1] == -1


def test_write_speed():
    import time

    obj = rig_with_bone()
    q = np.random.default_rng(0).normal(size=(600, 4))
    t = time.perf_counter()
    for _ in range(22):
        retarget.write(obj, np.arange(600), {"b": q})
    assert time.perf_counter() - t < 0.5


def evaluated(obj, path, frames, n):
    cb = retarget.channelbag(obj)
    return np.array([[cb.fcurves.find(path, index=i).evaluate(f) for i in range(n)] for f in frames])


def test_unbaked_keys_look_like_every_frame():
    obj = rig_with_bone()
    t = np.arange(1, 241)
    q = np.stack([np.ones_like(t) * 1.0, np.sin(t / 15) * 0.3, np.cos(t / 23) * 0.2, np.sin(t / 7) * 0.05], -1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    retarget.write(obj, t, {"b": q[:, [1, 2, 3, 0]]}, fit=(0.5, 0.5, [120], False, {}))
    path = 'pose.bones["b"].rotation_quaternion'
    kp = retarget.channelbag(obj).fcurves.find(path, index=0).keyframe_points
    assert len(kp) < 0.3 * len(t) and 120 in [k.co.x for k in kp], len(kp)
    assert {k.interpolation for k in kp} == {"BEZIER"} and {k.handle_left_type for k in kp} == {"ALIGNED"}
    err = np.degrees(keys.angle(evaluated(obj, path, t, 4), q)).max()
    assert err <= 0.5 + 1e-3, err


def test_same_frames_for_all_bones():
    arm = bpy.data.armatures.new("K2")
    obj = bpy.data.objects.new("K2", arm)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    for n in "ab":
        arm.edit_bones.new(n).tail = (0, 0, 1)
    bpy.ops.object.mode_set(mode="OBJECT")
    t = np.arange(100)
    qa = np.stack([np.sin(t / 9) * 0.3, t * 0, t * 0, np.ones_like(t) * 1.0], -1)
    qb = np.stack([t * 0, np.cos(t / 4) * 0.2, t * 0, np.ones_like(t) * 1.0], -1)
    basis = {n: q / np.linalg.norm(q, axis=1, keepdims=True) for n, q in (("a", qa), ("b", qb))}
    retarget.write(obj, t, basis, fit=(0.5, 0.5, [], True, {}))
    frames = [[k.co.x for k in retarget.channelbag(obj).fcurves.find(f'pose.bones["{n}"].rotation_quaternion',
                                                                  index=0).keyframe_points] for n in "ab"]  # fmt: skip
    assert frames[0] == frames[1] and len(frames[0]) < 50
