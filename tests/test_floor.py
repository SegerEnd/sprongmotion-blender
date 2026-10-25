import bpy
import numpy as np
from bl_ext.smtest.sprongmotion import blocks, generate, model


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
    obj.sprongmotion.floor_clamp = False
    generate.run_now(obj)
    assert lowest(obj, b).min() < -0.05, "the test needs a take that goes below the floor"
    obj.sprongmotion.floor_clamp = True
    generate.run_now(obj)
    assert not model.error(), model.error()
    assert lowest(obj, b).min() > -0.005
