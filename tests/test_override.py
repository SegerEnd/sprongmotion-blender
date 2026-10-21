import contextlib
import os
import tempfile

import bpy
from bl_ext.smtest.sprongmotion import blocks, data, engine, retarget
from test_rigs import build


def linked_rig():
    obj = build("simple16", junk=False)
    path = os.path.join(tempfile.mkdtemp(), "rig_anim.blend")
    bpy.data.libraries.write(path, {obj})
    obj_name = obj.name
    bpy.data.objects.remove(obj)
    with bpy.data.libraries.load(path, link=True) as (_, dst):
        dst.objects = [obj_name]
    bpy.context.collection.objects.link(dst.objects[0])
    return dst.objects[0]


def test_keys_go_onto_a_library_override():
    linked = linked_rig()
    assert "library override" in data.why_not_editable(linked)
    obj = linked.override_create(remap_local_usages=True)
    bpy.context.view_layer.objects.active = obj
    obj.sprongmotion.auto = False
    blocks.add(obj, "a person waves", 2)
    engine.run_now(obj)
    assert not engine.error(), engine.error()
    fc = retarget.channelbag(obj).fcurves.find('pose.bones["arm_up.L"].rotation_quaternion', index=1)
    assert fc is not None and len(fc.keyframe_points) > 2 and obj.animation_data.action.library is None


def test_a_linked_armature_without_override_gets_a_clear_message():
    obj = linked_rig()
    with contextlib.suppress(AttributeError, TypeError):
        obj.sprongmotion.auto = False
    engine.run_now(obj)
    assert "library override" in engine.error()


def test_freezing_on_a_library_override():
    obj = linked_rig().override_create(remap_local_usages=True)
    assert "make it a library override too" in data.freeze(obj, ["arm_up.L"]), "linked armature data: a message"
    obj.data.override_create(remap_local_usages=True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="POSE")
    for pb in obj.pose.bones:
        pb.select = pb.name == "arm_up.L"
    assert bpy.ops.sm.freeze() == {"FINISHED"}
    assert data.frozen(obj) == {"arm_up.L"}
    bpy.ops.object.mode_set(mode="OBJECT")
    path, name = os.path.join(tempfile.mkdtemp(), "shot.blend"), obj.name
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)
    assert data.frozen(bpy.data.objects[name]) == {"arm_up.L"}, "the frozen collection is saved with the override"
