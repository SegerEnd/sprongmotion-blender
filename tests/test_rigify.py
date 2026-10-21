import addon_utils
import bpy
import numpy as np
from bl_ext.smtest.sprongmotion import engine, retarget, rig
from test_retarget import fk, take

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
    target = retarget.Target(obj, engine.model().offsets)
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
