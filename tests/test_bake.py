import bpy
import numpy as np
from bl_ext.smtest.sprongmotion import blocks, engine, keys, retarget
from test_generate import setup


def test_one_action_per_block():
    obj = setup()
    engine.run_now(obj)
    timeline = obj.animation_data.action
    spans = [(b.text, b.end - b.start + 1) for b in blocks.settings(obj).blocks]
    obj.sprongmotion.key_density = "EXACT"
    exact = engine.block_actions(obj, [f"exact {t}" for t, _ in spans])
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
                worst = max(worst, np.degrees(keys.angle(np.array(q[0]), np.array(q[1]))).max())
        assert worst <= 0.51, f"{text}: {worst:.2f} degrees"


def test_block_actions_are_named_after_their_own_blocks():
    obj = setup()
    engine.run_now(obj)
    s = blocks.settings(obj)
    s.blocks[1].text = ""
    sits = s.blocks[2].end - s.blocks[2].start + 1
    assert bpy.ops.sm.bake("INVOKE_DEFAULT", actions="PER_BLOCK") == {"FINISHED"}
    made = {a.name: a for a in bpy.data.actions if a.use_fake_user}
    assert set(made) == {"A person walks forward", "A person sits down"}, set(made)
    assert tuple(made["A person sits down"].frame_range) == (1, sits)


def test_bake_without_a_model_says_so():
    obj = setup()
    prefs = engine.prefs()
    path, prefs.model_path = prefs.model_path, ""
    try:
        bpy.ops.sm.bake(actions="PER_BLOCK")
        raise AssertionError("the bake ran without a model")
    except RuntimeError as e:
        assert "Choose a SprongMotion model" in str(e), e
    finally:
        prefs.model_path = path
    assert len(blocks.settings(obj).blocks) == 3, "nothing baked, the blocks stay"
