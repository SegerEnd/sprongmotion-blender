import json
from types import SimpleNamespace

import bpy
import numpy as np
from bl_ext.smtest.sprongmotion import blocks, engine, lane, model, ops, timeline
from test_generate import all_codes, codes, setup
from undos import one_undo

Clip = SimpleNamespace(text="")  # background Blender has no clipboard
ops.Clipboard.get, ops.Clipboard.set = (lambda: Clip.text), (lambda t: setattr(Clip, "text", t))


def counting():
    """The model's block calls, counted as made blocks per call; undo with del m.blocks."""
    m, made = model.load(), []
    run = m.blocks

    def counted(texts, seconds, seeds, keep, *a):
        made.append(sum(k is None for k in keep))
        return run(texts, seconds, seeds, keep, *a)

    m.blocks = counted
    return made


def test_copy_paste_keeps_the_take_and_settings():
    obj = setup()
    s = blocks.settings(obj)
    obj.sprongmotion.variants, s.blocks[1].fresh, s.blocks[1].color = 2, True, (0.1, 0.2, 0.3)
    engine.run_now(obj)
    s.blocks[1].variant, s.blocks[1].locked, s.active = 1, True, 1
    assert bpy.ops.sm.block_copy() == {"FINISHED"} and json.loads(Clip.text)["take"]["variants"] == 2
    s.active = 2
    made = counting()
    try:
        assert bpy.ops.sm.block_paste() == {"FINISHED"}
        engine.run_now(obj)
    finally:
        del model.load().blocks
    new, src = s.blocks[3], s.blocks[1]
    assert s.active == 3 and not any(made), f"a paste of a kept take only decodes: {made}"
    assert np.array_equal(blocks.takes(new, model.load().id), blocks.takes(src, model.load().id))
    assert new.variant == 1 and np.array_equal(codes(new), codes(src))
    assert (new.text, new.seed, new.locked, new.fresh) == (src.text, src.seed, True, True)
    assert np.allclose(new.color, (0.1, 0.2, 0.3)) and new.end - new.start == src.end - src.start
    assert new.start == s.blocks[2].end and new.uid not in {b.uid for b in s.blocks[:3]}


def test_duplicate_shifts_later_blocks_and_makes_no_take():
    obj = setup()
    engine.run_now(obj)
    s = blocks.settings(obj)
    before, ends, clip = all_codes(obj), [(b.start, b.end) for b in s.blocks], Clip.text
    n = ends[0][1] - ends[0][0]
    made = counting()
    try:
        assert bpy.ops.sm.block_duplicate(index=0) == {"FINISHED"}
        engine.run_now(obj)
    finally:
        del model.load().blocks
    assert not any(made), made
    assert [(b.start, b.end) for b in s.blocks] == [ends[0], (ends[0][1], ends[0][1] + n)] + [
        (a + n, b + n) for a, b in ends[1:]
    ]
    assert s.active == 1 and Clip.text == clip, "duplicate leaves the clipboard"
    after = all_codes(obj)
    assert all(np.array_equal(a, b) for a, b in zip([before[0], *before], after, strict=True))


def test_paste_onto_another_rig_and_an_empty_lane():
    obj = setup()
    engine.run_now(obj)
    clip = blocks.to_clip(blocks.settings(obj).blocks[1])
    bpy.context.scene.frame_current = 40
    assert bpy.ops.sm.add_rig() == {"FINISHED"}
    other = bpy.context.active_object
    other.sprongmotion.auto = False
    Clip.text = clip
    assert bpy.ops.sm.block_paste() == {"FINISHED"}
    b = blocks.settings(other).blocks[0]
    assert len(blocks.settings(other).blocks) == 1 and b.start == 40 and not timeline.needs_take(b, model.load().id)
    assert np.array_equal(codes(b), codes(blocks.settings(obj).blocks[1]))
    engine.run_now(other)
    assert other.animation_data and other.animation_data.action, "the pasted take keys the other rig"


def test_garbage_on_the_clipboard_changes_nothing():
    obj = setup()
    engine.run_now(obj)
    s = blocks.settings(obj)
    good = json.loads(blocks.to_clip(s.blocks[0]))
    for junk in ("", "hello", "[1, 2]", '{"sprongmotion": "block"}', json.dumps({**good, "seconds": "x"}), "null"):
        Clip.text = junk
        assert bpy.ops.sm.block_paste() == {"CANCELLED"}, junk
        assert len(s.blocks) == 3 and all(not b.dirty for b in s.blocks)


def test_a_take_from_another_model_is_dropped():
    obj = setup()
    engine.run_now(obj)
    clip = json.loads(blocks.to_clip(blocks.settings(obj).blocks[0]))
    clip["take"]["model"] = "SprongMotion-14M@older"
    new = blocks.paste(obj, json.dumps(clip), 2)
    assert new.get("codes") is None and timeline.needs_take(new, model.load().id)
    engine.run_now(obj)
    assert blocks.settings(obj).blocks[3]["model"] == model.load().id


def test_each_is_one_undo_step():
    obj = setup()
    engine.run_now(obj)
    obj.sprongmotion.auto = True
    clip = json.loads(blocks.to_clip(blocks.settings(obj).blocks[1]))
    other = {**clip, "take": {**clip["take"], "model": "SprongMotion-14M@older"}}
    one_undo(lambda o: blocks.paste(o, blocks.to_clip(blocks.settings(o).blocks[0]), 0), "duplicate", makes=False)
    one_undo(lambda o: blocks.paste(o, json.dumps(clip), 2), "paste", makes=False)
    one_undo(lambda o: blocks.paste(o, json.dumps(other), 2), "paste from another model")


def test_the_keys_act_only_over_the_lane():
    obj = setup()
    engine.run_now(obj)
    y0, y1 = lane.rect()
    op = SimpleNamespace(action="duplicate")
    assert lane.SM_OT_lane_key.invoke(op, bpy.context, SimpleNamespace(mouse_region_y=y1 + 5)) == {"PASS_THROUGH"}
    assert lane.SM_OT_lane_key.invoke(op, bpy.context, SimpleNamespace(mouse_region_y=y0 + 1)) == {"FINISHED"}
    assert len(blocks.settings(obj).blocks) == 4
