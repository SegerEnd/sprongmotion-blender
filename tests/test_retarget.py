import addon_utils
import bpy
import numpy as np
from bl_ext.smtest.sprongmotion import model, retarget, rig

Z_UP = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])


def take(text="a person walks forward and waves with the right hand", seconds=3.0, seed=3):
    with model.load().blocks([text], [seconds], [seed], [None]) as (motion, _):
        return motion.rotations(), motion.pelvis()


def fk(rot, root):
    """Joint positions from the rotations on the reference skeleton, model frame."""
    return retarget.positions(retarget.globals_(rot), root, model.load().offsets)


def apply(obj, rot, root):
    rig.fill_map(obj)
    target = retarget.Target(obj, model.load().offsets)
    basis, loc = target.pose(rot, root)
    retarget.write(obj, np.arange(1, len(rot) + 1), basis, loc, target.map[0])
    return target


def posed(obj, frame):
    bpy.context.scene.frame_set(frame)
    m = obj.matrix_world
    return {pb.name: (m @ pb.head, m @ pb.tail) for pb in obj.pose.bones}


def own_rig():
    return rig.build_own_rig(bpy.context, model.load().offsets)


def place(obj, loc=(1.0, 2.0, 0.0), rot=(0, 0, 1.2), scale=0.5):
    obj.location, obj.rotation_euler, obj.scale = loc, rot, (scale,) * 3
    bpy.context.view_layer.update()


def test_rotated_scaled_own_rig_round_trip():
    bpy.context.scene.render.fps = 20
    rot, root = take()
    obj = own_rig()
    obj.sprongmotion.preset = "OWN"
    place(obj)
    target = apply(obj, rot, root)
    assert abs(target.ratio - 1) < 1e-6
    m, want = np.array(obj.matrix_world), fk(rot, root) @ Z_UP.T
    for f in range(0, len(rot), 7):
        got = np.array([posed(obj, f + 1)[n][0] for n in rig.JOINTS])
        exp = want[f] @ m[:3, :3].T + m[:3, 3]
        assert np.abs(got - exp).max() < 1e-4, f"frame {f}: {np.abs(got - exp).max() * 1000:.3f} mm"


DEF = {"l_hip": "DEF-thigh.L", "l_knee": "DEF-shin.L", "l_ankle": "DEF-foot.L", "r_hip": "DEF-thigh.R",
       "r_knee": "DEF-shin.R", "r_ankle": "DEF-foot.R", "l_shoulder": "DEF-upper_arm.L", "l_elbow": "DEF-forearm.L",
       "l_wrist": "DEF-hand.L", "r_shoulder": "DEF-upper_arm.R", "r_elbow": "DEF-forearm.R",
       "r_wrist": "DEF-hand.R"}  # fmt: skip


def angle(a, b):
    return np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))


def test_rigify_fk_round_trip():
    bpy.context.scene.render.fps = 20
    addon_utils.enable("rigify", default_set=True)
    bpy.ops.object.armature_human_metarig_add()
    bpy.ops.pose.rigify_generate()
    obj = bpy.context.active_object
    assert rig.auto_match(obj.data) == rig.match_preset(obj.data, "RIGIFY")
    rot, root = take()
    rig.fill_map(obj)
    rig.prepare(obj)
    target = retarget.Target(obj, model.load().offsets)
    basis, loc = target.pose(rot, root)
    retarget.write(obj, np.arange(1, len(rot) + 1), basis, loc, target.map[0])
    want, pb, worst, least = fk(rot, root) @ target.axes.T, obj.pose.bones, {}, {}
    for f in range(0, len(rot), 6):
        bpy.context.scene.frame_set(f + 1)
        for j, name in target.map.items():
            k = rig.CHILD[j]
            while k >= 0 and k not in target.map:
                k = rig.CHILD[k]
            if k >= 0:
                got = np.array(pb[target.map[k]].head - pb[name].head)
                a = angle(got, want[f, k] - want[f, j])
                worst[name], least[name] = max(worst.get(name, 0), a), min(least.get(name, 180), a)
            if k >= 0 and rig.JOINTS[j] in DEF and rig.JOINTS[k] in DEF:
                got = np.array(pb[DEF[rig.JOINTS[k]]].head - pb[DEF[rig.JOINTS[j]]].head)
                worst["DEF " + name] = max(worst.get("DEF " + name, 0), angle(got, want[f, k] - want[f, j]))
    torso = worst.pop("torso")
    assert torso < 5, f"pelvis to chest {torso:.2f} degrees (Rigify has no control for the first spine joint)"
    standing = {target.map[j] for j in retarget.STANDING if j in target.map}  # off by a fixed angle: standing rest
    assert max(v for n, v in worst.items() if n not in standing) < 0.1, worst
    assert all(least[n] > worst[n] - 0.1 for n in standing), (least, worst)
