import os
import tempfile

import addon_utils
import bpy
import numpy as np
from bl_ext.smtest.sprongmotion import blocks, generate, keyframes, model, retarget, rig, rig_data, takes
from test_generate import setup

PKG = "bl_ext.smtest.sprongmotion"
UI = {"window_managers", "workspaces", "screens", "all_ids"}


def inventory():
    """Every datablock in the file: (kind, name) -> users."""
    out = {}
    for kind in dir(bpy.data):
        coll = getattr(bpy.data, kind, None)
        if kind not in UI and isinstance(coll, bpy.types.bpy_prop_collection):
            out |= {(kind, i.name): i.users for i in coll if isinstance(i, bpy.types.ID)}
    return out


def add_on_props():
    found = []
    for kind, name in inventory():
        idb = getattr(bpy.data, kind)[name]
        system = idb.bl_system_properties_get() or {}
        found += [
            (kind, name, k) for k in [*idb.keys(), *system.keys()] if "sprong" in k.lower() or k in ("codes", "model")
        ]
    return found


def test_bake_and_remove_leave_plain_animation():
    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.context.scene.render.fps = 24
    bpy.ops.sm.add_rig()
    obj = bpy.context.active_object
    before = inventory()
    s = obj.sprongmotion
    s.auto, s.variants = False, 2
    blocks.add(obj, "a person walks forward", 2).seed = 3
    blocks.add(obj, "a person waves", 1.5).seed = 4
    generate.run_now(obj)
    bpy.ops.object.mode_set(mode="POSE")
    for pb in obj.pose.bones:
        pb.select = pb.name == "l_elbow"
    assert bpy.ops.sm.freeze() == {"FINISHED"}
    bpy.ops.object.mode_set(mode="OBJECT")
    saved = os.path.join(tempfile.mkdtemp(), "with-rig_data.blend")
    bpy.ops.wm.save_as_mainfile(filepath=saved, copy=True)
    assert add_on_props() and bpy.ops.sm.remove_data(every=True) == {"FINISHED"}
    assert not add_on_props(), add_on_props()
    after = inventory()
    assert set(after) - set(before) == {("actions", obj.animation_data.action.name)}, set(after) ^ set(before)
    assert not [k for k, users in after.items() if not users and before.get(k, 1)], "no new orphans"
    assert not obj.data.collections.get(rig_data.FROZEN) and {b.color.palette for b in obj.data.bones} == {"DEFAULT"}
    kinds = {k.type for fc in retarget.channelbag(obj).fcurves for k in fc.keyframe_points}
    assert kinds == {"KEYFRAME"}, kinds
    reference = rig.build_own_rig(bpy.context, model.load().offsets, name="Reference")
    assert list(reference.keys()) == list(obj.keys()) and [c.name for c in reference.data.collections] == [
        c.name for c in obj.data.collections
    ]


def test_without_the_add_on_the_file_opens_and_comes_back():
    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.sm.add_rig()
    obj = bpy.context.active_object
    obj.sprongmotion.auto = False
    blocks.add(obj, "a person walks forward", 2)
    path = os.path.join(tempfile.mkdtemp(), "data.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    prefs = bpy.context.preferences.addons[PKG].preferences
    paths = prefs.library_path, prefs.model_path
    addon_utils.disable(PKG)
    bpy.ops.wm.open_mainfile(filepath=path)
    assert "SprongMotion Rig" in bpy.data.objects
    addon_utils.enable(PKG, default_set=True)
    prefs = bpy.context.preferences.addons[PKG].preferences
    prefs.library_path, prefs.model_path = paths
    assert [b.text for b in blocks.settings(bpy.data.objects["SprongMotion Rig"]).blocks] == ["a person walks forward"]


def test_clear_motion_puts_the_animated_bones_back_to_rest():
    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.sm.add_rig()
    obj = bpy.context.active_object
    obj.sprongmotion.auto = False
    blocks.add(obj, "a person walks forward", 2)
    generate.run_now(obj)
    bpy.context.scene.frame_set(20)
    bpy.ops.object.mode_set(mode="POSE")
    for pb in obj.pose.bones:
        pb.select = pb.name == "l_elbow"
    assert bpy.ops.sm.freeze() == {"FINISHED"}, "its keys become the user's own"
    bpy.ops.object.mode_set(mode="OBJECT")
    elbow = obj.pose.bones["l_elbow"].rotation_quaternion.copy()
    assert obj.pose.bones["l_knee"].rotation_quaternion != (1, 0, 0, 0)
    assert bpy.ops.sm.remove_data(bake=False) == {"FINISHED"}
    pose = obj.pose.bones
    assert all(tuple(pose[n].rotation_quaternion) == (1, 0, 0, 0) for n in ("l_knee", "pelvis", "head"))
    assert tuple(pose["pelvis"].location) == (0, 0, 0) and pose["l_elbow"].rotation_quaternion == elbow


def test_one_action_per_block():
    obj = setup()
    generate.run_now(obj)
    timeline = obj.animation_data.action
    spans = [(b.text, b.end - b.start + 1) for b in blocks.settings(obj).blocks]
    obj.sprongmotion.key_density = "EXACT"
    exact = takes.block_actions(obj, [f"exact {t}" for t, _ in spans])
    assert bpy.ops.sm.bake(keys="0.5", actions="PER_BLOCK", asset=True) == {"FINISHED"}
    assert obj.animation_data.action == timeline, "the rig keeps the timeline"
    for (text, n), ref in zip(spans, exact, strict=True):
        action = bpy.data.actions[text.capitalize()]
        assert tuple(action.frame_range) == (1, n) and action.asset_data is not None
        worst = 0.0
        got, want = retarget.channelbag(obj, action).fcurves, retarget.channelbag(obj, ref).fcurves
        for fc in want:
            if "rotation_quaternion" in fc.data_path and not fc.array_index:
                q = [[[c.find(fc.data_path, index=i).evaluate(f) for i in range(4)] for f in range(1, n + 1)]
                     for c in (got, want)]  # fmt: skip
                worst = max(worst, np.degrees(keyframes.angle(np.array(q[0]), np.array(q[1]))).max())
        assert worst <= 0.51, f"{text}: {worst:.2f} degrees"


def test_block_actions_are_named_after_their_own_blocks():
    obj = setup()
    generate.run_now(obj)
    s = blocks.settings(obj)
    s.blocks[1].text = ""
    sits = s.blocks[2].end - s.blocks[2].start + 1
    assert bpy.ops.sm.bake("INVOKE_DEFAULT", actions="PER_BLOCK") == {"FINISHED"}
    made = {a.name: a for a in bpy.data.actions if a.use_fake_user}
    assert set(made) == {"A person walks forward", "A person sits down"}, set(made)
    assert tuple(made["A person sits down"].frame_range) == (1, sits)


def test_bake_without_a_model_says_so():
    obj = setup()
    prefs = model.prefs()
    path, prefs.model_path = prefs.model_path, ""
    try:
        bpy.ops.sm.bake(actions="PER_BLOCK")
        raise AssertionError("the bake ran without a model")
    except RuntimeError as e:
        assert "Choose a SprongMotion model" in str(e), e
    finally:
        prefs.model_path = path
    assert len(blocks.settings(obj).blocks) == 3, "nothing baked, the blocks stay"
