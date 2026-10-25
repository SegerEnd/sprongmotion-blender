import json
import os
import shutil
import tempfile

import bpy
import numpy as np
from bl_ext.smtest.sprongmotion import blocks, generate, model, retarget, rig_data
from test_cleanup import add_on_props
from test_generate import curve

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def fresh_rig(name="SprongMotion Rig"):
    bpy.context.scene.render.fps, bpy.context.scene.frame_current = 24, 1
    assert bpy.ops.sm.add_rig() == {"FINISHED"}, model.error()
    obj = bpy.context.active_object
    obj.name = name
    return obj


def texts(obj):
    s = blocks.settings(obj)
    return [b.text for b in s.blocks] if s else []


def test_a_motion_on_a_fresh_rig_makes_its_action_and_plays():
    obj = fresh_rig()
    assert obj.animation_data is None or obj.animation_data.action is None
    assert bpy.ops.sm.block_add(text="a person walks forward") == {"FINISHED"}
    action = obj.animation_data.action
    assert action.name == obj.name + "Action" and texts(obj) == ["a person walks forward"]
    assert len(curve(obj).keyframe_points) > 2 and not model.error(), "made at once and keyed"
    assert "blocks" not in obj.bl_system_properties_get()["sprongmotion"], "the blocks are the action's"


def test_each_action_keeps_its_own_blocks():
    obj = fresh_rig()
    blocks.add(obj, "a person walks forward", 2)
    walk, ad = obj.animation_data.action, obj.animation_data
    walk_codes, walk_keys = blocks.settings(obj).blocks[0]["codes"][:], len(curve(obj).keyframe_points)
    ad.action = bpy.data.actions.new("Wave")
    assert blocks.settings(obj) is None and blocks.has_blocks(bpy.context) is False, "a new action: an empty lane"
    blocks.add(obj, "a person waves", 1.5)
    assert texts(obj) == ["a person waves"] and len(ad.action.slots) == 1
    ad.action = walk
    assert texts(obj) == ["a person walks forward"] and blocks.settings(obj).blocks[0]["codes"][:] == walk_codes
    assert len(curve(obj).keyframe_points) == walk_keys, "the walk's keys are on its action as they were"
    assert texts(obj) != [b.text for b in bpy.data.actions["Wave"].sprongmotion.blocks]


def test_a_duplicated_action_copies_its_blocks_independently():
    obj = fresh_rig()
    blocks.add(obj, "a person walks forward", 2)
    original = obj.animation_data.action
    obj.animation_data.action = original.copy()
    copy = blocks.settings(obj).blocks[0]
    assert copy.text == "a person walks forward" and copy["codes"][:] == original.sprongmotion.blocks[0]["codes"][:]
    copy.text = "a person jumps"
    assert not model.error() and not copy.dirty and original.sprongmotion.blocks[0].text == "a person walks forward"
    assert copy["codes"][:] != original.sprongmotion.blocks[0]["codes"][:], "the copy's take is its own"


def test_an_action_on_two_rigs_is_keyed_on_each_with_its_own_slot():
    one = fresh_rig("One")
    blocks.add(one, "a person walks forward", 2)
    two = fresh_rig("Two")
    two.animation_data_create().action = one.animation_data.action
    bpy.context.view_layer.objects.active = one
    blocks.settings(one).blocks[0].text = "a person waves"
    assert texts(two) == ["a person waves"] and not model.error()
    for obj in (one, two):
        knee = retarget.channelbag(obj).fcurves.find('pose.bones["l_knee"].rotation_quaternion', index=1)
        assert knee is not None and len(knee.keyframe_points) > 2, obj.name


def test_switching_the_action_is_one_undo_step():
    obj = fresh_rig()
    blocks.add(obj, "a person walks forward", 2)
    bpy.ops.ed.undo_push(message="walk")
    ad = obj.animation_data
    ad.action = bpy.data.actions.new("Wave")
    blocks.add(obj, "a person waves", 1.5)
    bpy.ops.ed.undo_push(message="wave")
    bpy.ops.ed.undo()
    obj = bpy.data.objects["SprongMotion Rig"]
    assert obj.animation_data.action.name == "SprongMotion RigAction" and texts(obj) == ["a person walks forward"]
    bpy.ops.ed.redo()
    obj = bpy.data.objects["SprongMotion Rig"]
    assert obj.animation_data.action.name == "Wave" and texts(obj) == ["a person waves"]


def test_an_old_file_moves_the_rig_blocks_to_its_action_bit_for_bit():
    """old-blocks.blend: saved by an add-on version that kept blocks on the rig (Walker: two made blocks with two takes;
    Idle: Loop and a block without a take, so no action yet). old-blocks.json: what it held."""
    with open(os.path.join(DATA, "old-blocks.json")) as f:
        want = json.load(f)
    path = os.path.join(tempfile.mkdtemp(), "old.blend")
    shutil.copy(os.path.join(DATA, "old-blocks.blend"), path)
    bpy.ops.wm.open_mainfile(filepath=path)
    for _ in range(2):  # and once more after saving: moved once, nothing lost
        for name in ("Walker", "Idle"):
            obj = bpy.data.objects[name]
            s, rig = blocks.settings(obj), want[name + "/rig"]
            assert s.id_data.name == name + "Action", s.id_data.name
            assert (s.loop, s.active, s.next_uid) == (rig["loop"], rig["active"], rig["next_uid"])
            assert obj.sprongmotion.variants == rig["variants"]
            for b, w in zip(s.blocks, want[name], strict=True):
                keys = ("text", "start", "end", "variant", "variants", "in_place", "fresh", "seed")
                assert [getattr(b, k) for k in keys] == [w[k] for k in keys]
                assert np.allclose(b.color, w["color"]) and b.get("model") == w["model"]
                assert (list(b["codes"]) if "codes" in b else None) == w["codes"], "the takes bit for bit"
            assert not set(obj.bl_system_properties_get()["sprongmotion"]) & {"blocks", "loop", "active"}
        bpy.ops.wm.save_mainfile()
        bpy.ops.wm.open_mainfile(filepath=path)
    obj = bpy.data.objects["Walker"]
    generate.run_now(obj)
    assert not model.error() and not any(b.dirty for b in blocks.settings(obj).blocks), "the takes decode as is"


def test_remove_takes_the_playing_action_then_the_whole_file():
    obj = fresh_rig()
    obj.sprongmotion.auto = False
    blocks.add(obj, "a person walks forward", 2)
    walk = obj.animation_data.action
    obj.animation_data.action = bpy.data.actions.new("Wave")
    blocks.add(obj, "a person waves", 1.5)
    generate.run_now(obj)
    assert bpy.ops.sm.remove_data(bake=False) == {"FINISHED"}
    assert "Wave" not in bpy.data.actions, "left empty: it goes"
    assert texts(obj) == [] and blocks.stored(walk) and blocks.stored(obj), "the walk keeps its blocks and the rig"
    obj.animation_data_create().action = walk
    assert texts(obj) == ["a person walks forward"]
    assert bpy.ops.sm.remove_data(every=True) == {"FINISHED"}
    assert not add_on_props(), add_on_props()


def test_a_linked_action_with_blocks_is_read_only():
    obj = fresh_rig()
    obj.sprongmotion.auto = False
    blocks.add(obj, "a person walks forward", 2)
    action = obj.animation_data.action
    path, name = os.path.join(tempfile.mkdtemp(), "anim.blend"), action.name
    bpy.data.libraries.write(path, {action})
    obj.animation_data.action = None
    bpy.data.actions.remove(action)
    with bpy.data.libraries.load(path, link=True) as (_, dst):
        dst.actions = [name]
    obj.animation_data.action = dst.actions[0]
    assert texts(obj) == ["a person walks forward"], "shown"
    assert "read-only" in rig_data.why_not_editable(obj)
    assert not bpy.ops.sm.block_add.poll() and not bpy.ops.sm.block_remove.poll()
