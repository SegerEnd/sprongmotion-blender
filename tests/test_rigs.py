import bpy
import numpy as np
import rigs
from bl_ext.smtest.sprongmotion import blocks, engine, retarget, rig
from mathutils import Matrix, Quaternion, Vector
from test_retarget import apply, fk, posed, take
from test_takes import slide


def build(name, junk=True):
    spec = rigs.RIGS[name]
    obj = bpy.data.objects.new(name, bpy.data.armatures.new(name))
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    eb = obj.data.edit_bones
    for n, (_, head, tail) in rigs.bones(spec, junk).items():
        e = eb.new(n)
        e.head, e.tail = head, tail
    for n, (parent, *_) in rigs.bones(spec, junk).items():
        eb[n].parent = eb[parent] if parent else None
    bpy.ops.object.mode_set(mode="OBJECT")
    turn, size = Quaternion(spec.get("rotation", (1, 0, 0, 0))), (spec.get("scale", 1),) * 3
    obj.matrix_world = Matrix.LocRotScale(Vector(spec.get("location", (0, 0, 0))), turn, size)
    bpy.context.view_layer.update()
    return obj


def check(name):
    """Mapping, every mapped bone pointing where the model's joints go, junk untouched, and with knees the leg IK."""
    obj = build(name)
    bpy.context.scene.render.fps = 20
    rot, root = take(seed=4)
    target = apply(obj, rot, root)
    assert {rig.JOINTS[j]: b for j, b in target.map.items()} == rigs.expected(rigs.RIGS[name])
    want, m = fk(rot, root) @ target.axes.T, np.array(obj.matrix_world)[:3, :3]
    off = {}  # the neck and feet keep the rig's standing rest: a fixed angle off the model's level rest line
    for b, j in target.joint_of.items():
        rest, model = target._dirs(b, j)
        turned = retarget.qrot(target.align[b], rest)
        cos = turned.dot(model) / np.linalg.norm(turned) / np.linalg.norm(model)
        off[j] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    for f in range(0, len(rot), 8):
        pose = posed(obj, f + 1)
        for j, k in target.ends.items():
            name_j = target.map[j]
            got = pose[target.map[k]][0] - pose[name_j][0] if k in target.map else pose[name_j][1] - pose[name_j][0]
            d = m @ (want[f, k] - want[f, j])
            angle = np.degrees(
                np.arccos(np.clip(np.array(got).dot(d) / np.linalg.norm(got) / np.linalg.norm(d), -1, 1))
            )
            assert abs(angle - off[j]) < 0.1, f"{name_j} frame {f}: {angle:.3f} degrees, want {off[j]:.3f}"
    junk = set(rigs.RIGS[name].get("junk", {}))
    assert not junk & {fc.data_path.split('"')[1] for fc in retarget.channelbag(obj).fcurves}
    assert all(tuple(obj.pose.bones[n].rotation_quaternion) == (1, 0, 0, 0) for n in junk)
    if 4 in target.map:
        assert legs_hold(obj, target)


def legs_hold(obj, target):
    """Planted ankles slide far less with foot lock (and its leg IK) than without."""
    obj.animation_data_clear()
    s = obj.sprongmotion
    s.auto = False
    blocks.add(obj, "a person walks forward", 3).seed = 5
    out = []
    for lock in (False, True):
        s.foot_lock = lock
        engine.run_now(obj)
        with engine.timeline(obj) as (motion, segs):
            contacts = motion.contacts()
        frames, times = engine.timing(segs)
        planted = contacts[np.round(times).astype(int).clip(0, len(contacts) - 1), 0]
        pos = []
        for f in frames:
            bpy.context.scene.frame_set(int(f))
            pos.append(np.array(obj.matrix_world @ obj.pose.bones[target.map[7]].head))
        out.append(slide(np.array(pos), planted))
    return out[1] < 0.3 * out[0] or print(f"> slide without / with foot lock: {out}")


for _name in rigs.RIGS:
    globals()[f"test_{_name}"] = lambda _name=_name: check(_name)
