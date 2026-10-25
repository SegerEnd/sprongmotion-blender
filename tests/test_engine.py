import os
import tempfile
from types import SimpleNamespace

import bpy
import numpy as np
from bl_ext.smtest.sprongmotion import __package__ as pkg
from bl_ext.smtest.sprongmotion import blocks, generate, model, panels, retarget, timeline_motion

PREFS = bpy.context.preferences.addons[pkg].preferences
LIB, MODEL = PREFS.library_path, PREFS.model_path


def rig_with_block(text="a person walks forward", auto=True):
    bpy.ops.sm.add_rig()
    obj = bpy.context.active_object
    obj.sprongmotion.auto = auto
    blocks.add(obj, text, 2)
    return obj


def keys(obj):
    fc = retarget.channelbag(obj).fcurves.find('pose.bones["l_knee"].rotation_quaternion', index=1)
    return [k.co.y for k in fc.keyframe_points] if fc else []


def test_plain_messages_for_missing_or_wrong_files():
    PREFS.library_path = ""
    assert model.load() is None and "Choose the SprongMotion library" in model.error()
    loaded, model._state.lib = model._state.lib, None
    PREFS.library_path = "/usr/lib/libz.dylib"
    assert model.load() is None and "Cannot load the library" in model.error()
    model._state.lib = loaded
    PREFS.library_path = LIB
    bad = os.path.join(tempfile.mkdtemp(), "bad.gguf")
    with open(bad, "wb") as f:
        f.write(b"GGUF" + bytes(100))
    PREFS.model_path = bad
    assert model.load() is None and "Cannot load the model" in model.error() and bad in model.error()
    PREFS.model_path = MODEL
    assert model.load() is not None and not model.error()
    PREFS.library_path = LIB + ".other"
    assert model.load() is None and "Restart Blender" in model.error()
    PREFS.library_path = LIB
    assert model.load() is not None


def test_engine_recovers_from_an_exception():
    obj = rig_with_block(auto=False)
    write = retarget.write
    retarget.write = lambda *a, **k: 1 / 0
    try:
        generate.run_now(obj)
    finally:
        retarget.write = write
    assert "division by zero" in model.error() and not generate.busy()
    generate.run_now(obj)
    assert not model.error() and keys(obj)


def test_bad_codes_and_a_missing_bone_fail_safe():
    obj = rig_with_block(auto=False)
    generate.run_now(obj)
    b = blocks.settings(obj).blocks[0]
    b["codes"] = [9999] * len(b["codes"])
    generate.run_now(obj)
    assert "code" in model.error() and not generate.busy()
    blocks.mark_dirty(obj, b.uid)
    generate.run_now(obj)
    assert not model.error()
    obj.sprongmotion.bones[1].bone = "no such bone"
    obj.sprongmotion.bones[0].bone = "no such bone"
    generate.run_now(obj)
    assert not model.error()


def test_undo_during_a_take_cancels_it():
    obj = rig_with_block(auto=False)
    obj.sprongmotion.variants = 2
    generate.edited(obj, force=True)
    generate._state.due[obj.session_uid] = 0
    generate._tick()
    assert generate.busy(obj)
    for handler in bpy.app.handlers.undo_post:
        handler(bpy.context.scene, None)
    assert not generate.busy() and blocks.settings(obj).blocks[0].variants == 0, "no take stored after the undo"


def test_deleting_the_armature_or_loading_a_file_mid_take():
    obj = rig_with_block(auto=False)
    obj.sprongmotion.variants = 2
    generate.edited(obj, force=True)
    generate._state.due[obj.session_uid] = 0
    generate._tick()
    bpy.data.objects.remove(obj)
    while generate._tick():
        pass
    assert not generate.busy() and not model.error()
    obj = rig_with_block(auto=False)
    obj.sprongmotion.variants = 2
    generate.edited(obj, force=True)
    generate._state.due[obj.session_uid] = 0
    generate._tick()
    bpy.ops.wm.read_homefile(use_empty=True)
    assert not generate.busy() and generate._tick() is None


def test_codes_from_another_model_are_regenerated():
    obj = rig_with_block(auto=False)
    generate.run_now(obj)
    b = blocks.settings(obj).blocks[0]
    b["model"] = "SprongMotion-14M@older"
    assert timeline_motion.needs_take(b, model.load().id)
    generate.run_now(obj)
    assert b["model"] == model.load().id


def test_a_small_edit_and_its_take_are_one_undo_step():
    obj = rig_with_block()
    pushes = []
    push, generate.undo_push = generate.undo_push, pushes.append
    try:
        before = keys(obj)
        assert before and not blocks.settings(obj).blocks[0].dirty, "the take came with the add"
        bpy.ops.sm.block_regenerate(new_seed=True)
        assert keys(obj) != before and not generate.busy(), "the operator's own undo step holds the new take"
        obj.sprongmotion.variants = 2
        bpy.ops.sm.block_regenerate(new_seed=True)
        assert generate.busy() or generate._state.due, "a big job goes to the timer"
        while generate._tick():
            pass
        assert pushes == ["SprongMotion take"]
    finally:
        generate.undo_push = push
    assert np.isfinite(keys(obj)).all()


class _Any:
    """A layout that takes any call."""

    def __getattr__(self, _name):
        return lambda *a, **k: self


def test_draws_load_nothing():
    obj = rig_with_block(auto=False)
    blocks.settings(obj).blocks[0].text = "a person does a pirouettez"
    assert model.unknown_words("a person does a pirouettez", look=False), "the edit looked the words up"
    model.unload()
    for draw in (panels.draw_main, panels.draw_rig, panels.draw_output):
        draw(_Any(), bpy.context)
    type(PREFS).draw(SimpleNamespace(layout=_Any(), after_block="HOLD"), bpy.context)
    assert model.loaded() is None, "a draw loaded the model"
    assert model.load() is not None and model.loaded() is not None


def test_one_cmd_z_undoes_an_edit_and_its_take_and_nothing_regenerates():
    from undos import edits, one_undo

    obj = rig_with_block(auto=False)
    blocks.add(obj, "a person waves", 1.5)
    generate.run_now(obj)
    obj.sprongmotion.auto = True
    for label, edit in edits():
        one_undo(edit, label)
