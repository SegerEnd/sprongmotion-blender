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
