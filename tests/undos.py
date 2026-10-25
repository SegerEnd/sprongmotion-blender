"""Edits through Blender's real undo stack: what one Cmd+Z and one redo give back, with the timers run as Blender
would run them."""

import bpy
import numpy as np
from bl_ext.smtest.sprongmotion import blocks, engine, model, paths, retarget

NAME = "SprongMotion Rig"


def state():
    obj = bpy.data.objects[NAME]
    s, m = blocks.settings(obj), model.load()
    fc = retarget.channelbag(obj).fcurves.find('pose.bones["l_knee"].rotation_quaternion', index=1)
    keys = np.array([k.co for k in fc.keyframe_points]).round(6).tolist() if fc else []
    codes = [None if (t := blocks.codes(b, m.id)) is None else t.tolist() for b in s.blocks]
    return [b.end - b.start for b in s.blocks], codes, keys


def run_timers():
    """Every add-on timer until none is due, at once (background Blender runs no timers)."""
    for _ in range(1000):
        due = [f for f in (engine._tick, paths._tick, engine._settle) if bpy.app.timers.is_registered(f)]
        if not due:
            return
        for f in due:
            bpy.app.timers.unregister(f)
            engine._state.due.update(dict.fromkeys(engine._state.due, 0))
            paths._state.due.update(dict.fromkeys(paths._state.due, 0))
            if f() is not None:
                bpy.app.timers.register(f, first_interval=0)


def one_undo(edit, label, makes=True):
    """edit(obj) as the UI does it (its own undo step after it), the timers, then Cmd+Z and redo: asserts one Cmd+Z
    gives back the state before and nothing is made after it, and redo the edit's result with nothing made. makes:
    the edit makes a take (else it only decodes)."""
    run_timers()
    bpy.ops.ed.undo_push(message="before")
    before = state()
    m, made, push = model.load(), [], engine.undo_push
    run = m.blocks

    def counted(texts, seconds, seeds, keep, *a):
        made.append(sum(k is None for k in keep))
        return run(texts, seconds, seeds, keep, *a)

    m.blocks, engine.undo_push = counted, lambda message: bpy.ops.ed.undo_push(message=message)
    try:
        edit(bpy.data.objects[NAME])
        bpy.ops.ed.undo_push(message=label)
        run_timers()
        after = state()
        assert after != before and any(made) == makes, f"{label}: the edit made a take: {made}, expected {makes}"
        made.clear()
        bpy.ops.ed.undo()
        run_timers()
        assert state() == before, f"{label}: one Cmd+Z lands at {state()[0]}, before was {before[0]}"
        assert not any(made), f"{label}: something was made after the undo: {made}"
        bpy.ops.ed.redo()
        run_timers()
        assert state() == after and not any(made), f"{label}: redo gives the edit back, made {made}"
    finally:
        del m.blocks
        engine.undo_push = push


def edits():
    """(label, edit) for a typed Seconds value, a lane edge drag and a text edit on block 0."""

    def typed(obj):
        blocks.settings(obj).blocks[0].seconds += 0.5

    def drag(obj):  # what SM_OT_lane_click does: resize on every mouse move, mark dirty on release
        b = blocks.settings(obj).blocks[0]
        for end in range(b.end + 1, b.end + 9):
            blocks.resize(obj, 0, end=end)
        blocks.mark_dirty(obj, b.uid)

    def text(obj):
        blocks.settings(obj).blocks[0].text = "a person walks forward slowly"

    return [("typed seconds", typed), ("edge drag", drag), ("text edit", text)]
