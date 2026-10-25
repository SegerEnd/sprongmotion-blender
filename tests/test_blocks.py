import os
import time

import bpy
import numpy as np
from bl_ext.smtest.sprongmotion import blocks, engine, model, takes, timeline
from test_generate import all_codes, curve, setup


def sampled(obj, frames):
    fc = curve(obj)
    return np.array([fc.evaluate(f) for f in frames])


def test_split_keeps_the_motion():
    obj = setup()
    obj.sprongmotion.key_density = "EXACT"
    engine.run_now(obj)
    s = blocks.settings(obj)
    frames, tokens = range(s.blocks[0].start, s.blocks[-1].end), s.blocks[0].tokens
    before = sampled(obj, frames)
    s.blocks[0].locked = True
    assert bpy.ops.sm.block_split(index=0, frame=s.blocks[0].start + 24) == {"FINISHED"}
    engine.run_now(obj)
    a, b = s.blocks[0], s.blocks[1]
    assert (a.tokens, b.tokens) == (tokens // 2, tokens // 2) and a.locked and b.locked
    assert np.abs(sampled(obj, frames) - before).max() < 1e-4, "both halves show what the block showed"


def test_one_block_per_timer_tick_same_takes():
    obj = setup()
    m, calls, run = model.load(), [], model.load().blocks

    def counted(texts, seconds, seeds, keep, *a):
        calls.append(sum(k is None for k in keep))
        return run(texts, seconds, seeds, keep, *a)

    job, m.blocks = engine._Job(obj.session_uid, push=False), counted
    try:
        while not job.advance():
            pass
    finally:
        del m.blocks
    assert calls == [2, 2, 1], "one block per call, a short stand-in when a block to be made follows, no extra decode"
    made = all_codes(obj)
    for b in blocks.settings(obj).blocks:
        b.dirty = True
    engine.run_now(obj)
    assert all((a == b).all() for a, b in zip(made, all_codes(obj), strict=True)), "the same takes as one call"


def test_edit_to_keys_time():
    """One 2 s block edited in a 6-block timeline. Reported; the add-on's own part (the keys) is held to a budget in
    make test-slow, where nothing runs beside it; the model's part is the runtime's."""
    obj = setup()
    for text in ("a person waves", "a person runs", "a person turns around"):
        blocks.add(obj, text, 2)
    engine.run_now(obj)
    obj.sprongmotion.auto = True
    spent, apply = [], takes.apply

    def timed(*a, **k):
        t0 = time.perf_counter()
        apply(*a, **k)
        spent.append(time.perf_counter() - t0)

    takes.apply, total = timed, []
    try:
        for k in range(3):
            t0 = time.perf_counter()
            blocks.settings(obj).blocks[2].text = "a person sits down" + " slowly" * (k % 2)
            total.append(time.perf_counter() - t0)
    finally:
        takes.apply = apply
    edit, keys = 1000 * min(total), 1000 * min(spent)
    threads = model.THREADS or "all"
    print(f"> edit to keys {edit:.0f} ms: model {edit - keys:.0f} ms, keys {keys:.0f} ms ({threads} model threads)")
    assert not os.environ.get("SM_SLOW") or keys < 60, f"keys took {keys:.0f} ms"


def test_a_take_belongs_to_its_text():
    """A block whose text does not match its take (an undo step, a text written past the update) is made again."""
    obj = setup()
    blocks.add(obj, "a person walks", 2)
    engine.run_now(obj)
    b = blocks.settings(obj).blocks[0]
    before = blocks.codes(b, model.load().id).copy()
    b["text"] = "a person stands still"  # the stored text alone: no update, so no dirty mark
    assert not b.dirty and timeline.needs_take(b, model.load().id)
    engine.run_now(obj)
    assert not timeline.needs_take(b, model.load().id) and not np.array_equal(before, blocks.codes(b, model.load().id))
