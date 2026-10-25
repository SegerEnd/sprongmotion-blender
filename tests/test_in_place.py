import json
import os
import tempfile

import bpy
import numpy as np
import undos
from bl_ext.smtest.sprongmotion import blocks, generate, model, takes
from test_copy import counting
from test_generate import new_rig


def setup(auto=False):
    obj = new_rig(auto)
    for k, text in enumerate(("a person walks forward", "a person walks forward", "a person walks forward quickly")):
        blocks.add(obj, text, 2).seed = k + 1  # random seeds: a walk may barely move
    obj.sprongmotion.key_density = "EXACT"
    generate.run_now(obj)
    return obj


def pelvis(obj, frames):
    heads = []
    for f in frames:
        bpy.context.scene.frame_set(int(f))
        heads.append(np.array(obj.pose.bones["pelvis"].head)[:2])
    return np.array(heads)


def step(p, at):
    return np.linalg.norm(p[at] - p[at - 1])


def test_a_block_in_place_between_travelling_blocks():
    obj = setup(auto=True)
    s = blocks.settings(obj)
    a, b, c = s.blocks
    made = counting()
    try:
        b.in_place = True
    finally:
        del model.load().blocks
    assert not any(made), f"toggling only decodes: {made}"
    p = pelvis(obj, range(a.start, c.end))
    mid = p[b.start - a.start : b.end - a.start]
    walk = np.median(np.linalg.norm(np.diff(p[: a.end - a.start], axis=0), axis=1))
    # its last frame samples between the block and the next one, so it may start that block's first step
    assert np.ptp(mid, axis=0).max() < 0.25 * walk, f"in place: {np.ptp(mid, axis=0)}, the walk {walk:.3f} m a frame"
    assert np.ptp(p[: a.end - a.start], axis=0).max() > 0.5 and np.ptp(p[c.start - a.start :], axis=0).max() > 0.5
    for at in (b.start - a.start, c.start - a.start):  # no jump at either seam: no step beyond the walk's own
        own = max(step(p, at - 1), step(p, at + 1))
        assert step(p, at) < own + 0.01, f"seam at {at}: {step(p, at):.3f} m, the walk's own step {own:.3f} m"


def test_toggling_is_one_undo_step():
    setup(auto=True)

    def state():
        obj = bpy.data.objects[undos.NAME]
        return [b.in_place for b in blocks.settings(obj).blocks], pelvis(obj, range(1, 140, 7)).round(5).tolist()

    old, undos.state = undos.state, state
    try:
        undos.one_undo(lambda o: setattr(blocks.settings(o).blocks[1], "in_place", True), "in place", makes=False)
        undos.one_undo(lambda o: setattr(blocks.settings(o), "in_place", True), "rig-wide in place", makes=False)
    finally:
        undos.state = old


def test_the_rig_wide_option_sets_all_blocks():
    obj = setup(auto=True)
    s = blocks.settings(obj)
    s.in_place = True
    assert s.in_place and all(b.in_place for b in s.blocks)
    assert np.ptp(pelvis(obj, range(1, s.blocks[-1].end, 5)), axis=0).max() < 0.01
    assert blocks.add(obj, "a person jumps", 1).in_place, "a block added to a rig all in place is in place"
    s.blocks[0].in_place = False
    assert not s.in_place, "mixed blocks show it off"
    s.in_place = False
    assert not any(b.in_place for b in s.blocks)


def test_copy_paste_and_split_carry_it():
    obj = setup()
    s = blocks.settings(obj)
    s.blocks[1].in_place = True
    clip = blocks.to_clip(s.blocks[1])
    assert json.loads(clip)["in_place"]
    assert blocks.paste(obj, clip, 2).in_place and not blocks.paste(obj, blocks.to_clip(s.blocks[0]), 0).in_place
    assert blocks.split(obj, 2, (s.blocks[2].start + s.blocks[2].end) // 2).in_place


def test_per_block_bake_respects_it():
    obj = setup()
    s = blocks.settings(obj)
    s.blocks[1].in_place = True
    walk, still, _ = takes.block_actions(obj, ["walk", "still", "quick"])
    for action, moves in ((walk, True), (still, False)):
        obj.animation_data.action = action
        travel = np.ptp(pelvis(obj, range(1, int(action.frame_range[1]) + 1)), axis=0).max()
        assert (travel > 0.5) if moves else (travel < 0.01), f"{action.name}: {travel:.3f} m"


def test_an_old_file_with_the_rig_wide_option_loads_all_in_place():
    obj = setup()
    obj.sprongmotion["in_place"] = True  # how older files store a rig-wide In Place
    path = os.path.join(tempfile.mkdtemp(), "old.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)
    s = blocks.settings(bpy.data.objects[undos.NAME])
    assert all(b.in_place for b in s.blocks) and s.in_place and "in_place" not in s
