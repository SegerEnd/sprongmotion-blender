from types import SimpleNamespace

import bpy
from bl_ext.smtest.sprongmotion import block_text, blocks, operators, timeline_lane


def key(type_, unicode="", ctrl=False):
    return SimpleNamespace(type=type_, unicode=unicode, ctrl=ctrl, oskey=False)


def test_typing_in_place():
    text, caret = "", 0
    for ch in "a person runs":
        text, caret = block_text.type_key(text, caret, key(ch.upper(), ch))
    assert (text, caret) == ("a person runs", 13)
    text, caret = block_text.type_key(text, caret, key("HOME"))
    text, caret = block_text.type_key(text, caret, key("DEL"))
    assert (text, caret) == (" person runs", 0)
    text, caret = block_text.type_key(text, caret, key("V", ctrl=True), "slowly\nmore")
    assert (text, caret) == ("slowly person runs", 6)
    assert block_text.type_key(text, caret, key("LEFT_ARROW")) == (text, 5)
    assert block_text.type_key(text, caret, key("BACK_SPACE")) == ("slowl person runs", 5)
    assert block_text.type_key(text, caret, key("A", "a", ctrl=True)) == (text, caret)


class _Region:
    """Frame f at x = 10 * f, like region.view2d."""

    width = 2000
    view2d = SimpleNamespace(
        view_to_region=lambda f, y, clip=False: (10 * f, y), region_to_view=lambda x, y: (x / 10, y)
    )


def test_add_zone_is_the_drawn_chip():
    obj = bpy.data.objects.new("L", bpy.data.armatures.new("L"))
    bpy.context.scene.collection.objects.link(obj)
    obj.sprongmotion.auto = False
    s = None
    y = sum(timeline_lane.rect()) / 2
    bpy.context.scene.frame_current = 5
    x0, x1, label = timeline_lane.add_chip(_Region, s)
    assert label == "+ Add motion" and x0 == 50 and timeline_lane.hit(_Region, s, x0 + 1, y) == ("add", -1)
    s = blocks.settings(obj, create=True)
    b = s.blocks.add()
    b.start, b.end = 0, 30
    x0, x1, _ = timeline_lane.add_chip(_Region, s)
    assert timeline_lane.hit(_Region, s, x1 + 50, y) == (None, -1), "the lane right of the chip adds nothing"
    assert timeline_lane.hit(_Region, s, (x0 + x1) / 2, y) == ("add", -1)
    assert timeline_lane.hit(_Region, s, 150, y) == ("body", 0)
    assert timeline_lane.hit(_Region, s, 300, y) == ("right", 0)


def test_operators_exist():
    names = {i for i, *_ in timeline_lane.KEYS} | {c.bl_idname for c in operators.CLASSES if hasattr(c, "bl_idname")}
    for n in names:
        group, op = n.split(".")
        assert hasattr(bpy.types, f"{group.upper()}_OT_{op}"), n


class _Wide:
    """2 px per frame: SNAP reaches 4 frames."""

    view2d = SimpleNamespace(view_to_region=lambda f, y, clip=False: (2 * f, y), region_to_view=lambda x, y: (x / 2, y))


def _two_blocks():
    scene = bpy.context.scene
    scene.render.fps, scene.render.fps_base, scene.frame_current = 24, 1.0, 300
    obj = bpy.data.objects.new("S", bpy.data.armatures.new("S"))
    scene.collection.objects.link(obj)
    obj.sprongmotion.auto = False
    s = blocks.settings(obj, create=True)
    for k in range(2):
        b = s.blocks.add()
        b.uid, b.start, b.end = k + 1, 48 * k, 48 * k + 48
    return scene, obj


def _drag(obj, kind, frame, ctrl=False, uid=1, shift=False, f0=10.0, alt=False):
    """Drag from a fresh start to frame: block uid's kind ('right', or 'body' with shift); the blocks' (start, end)."""
    s = blocks.settings(obj)
    for k, b in enumerate(s.blocks):
        b.start, b.end = 48 * k, 48 * k + 48
    snapshot = [(b.uid, b.start, b.end, b.locked, b.dirty, b.remake) for b in s.blocks]
    op = SimpleNamespace(kind=kind, uid=uid, shift=shift, snapshot=snapshot, f0=f0, region=_Wide)
    timeline_lane.SM_OT_lane_click.drag(op, obj, 2 * frame, ctrl, alt)
    return [(b.start, b.end) for b in s.blocks]


def test_snap_off_is_whole_frames():
    scene, obj = _two_blocks()
    scene.tool_settings.use_snap_anim, scene.tool_settings.snap_anim_element = False, "SECOND"
    assert _drag(obj, "right", 70.6)[0] == (0, 71)
    assert _drag(obj, "body", 31, shift=True)[0] == (21, 69)


def test_snap_to_seconds():
    scene, obj = _two_blocks()
    ts = scene.tool_settings
    ts.use_snap_anim, ts.snap_anim_element = True, "SECOND"
    assert _drag(obj, "right", 70.6)[0] == (0, 72), "near a second boundary: on it"
    assert _drag(obj, "right", 66.4)[0] == (0, 66), "far from one: whole frames"
    assert _drag(obj, "body", 31, shift=True) == [(24, 72), (72, 120)], "Shift-move puts the nearest edge on it"
    assert _drag(obj, "body", 22, shift=True)[0] == (12, 60)


def test_ctrl_inverts():
    scene, obj = _two_blocks()
    ts = scene.tool_settings
    ts.use_snap_anim, ts.snap_anim_element = True, "SECOND"
    assert _drag(obj, "right", 70.4, ctrl=True)[0] == (0, 70)
    ts.use_snap_anim = False
    assert _drag(obj, "right", 70.4, ctrl=True)[0] == (0, 72)


def test_snap_to_marker_playhead_and_blocks():
    scene, obj = _two_blocks()
    ts = scene.tool_settings
    ts.use_snap_anim, ts.snap_anim_element = True, "MARKER"
    scene.timeline_markers.new("M", frame=60)
    assert _drag(obj, "right", 58.3)[0] == (0, 60), "nearest marker"
    ts.snap_anim_element = "FRAME"
    assert _drag(obj, "right", 58.3)[0] == (0, 58), "markers only with Snap To Nearest Marker"
    scene.frame_current = 55
    assert _drag(obj, "right", 53.2)[0] == (0, 55), "the playhead"
    assert _drag(obj, "right", 93.5) == [(0, 96), (96, 144)], "another block's edge"


def test_alt_drag_picks_a_range():
    scene, obj = _two_blocks()
    scene.tool_settings.use_snap_anim = True
    context = SimpleNamespace(
        active_object=obj, region=_Wide, window_manager=SimpleNamespace(modal_handler_add=lambda op: None)
    )
    for alt, kind in ((False, "body"), (True, "range")):
        op = SimpleNamespace(hit=lambda c, e: ("body", 0))
        event = SimpleNamespace(alt=alt, ctrl=False, shift=False, mouse_region_x=20)
        assert timeline_lane.SM_OT_lane_click.invoke(op, context, event) == {"RUNNING_MODAL"} and op.kind == kind
    timeline_lane.SM_OT_lane_click.drag(op, obj, 2 * 30.6)
    assert timeline_lane._state.range == (1, 10, 31)
    timeline_lane._state.range = None


def test_an_edge_drag_ripples_the_run_and_alt_does_not():
    scene, obj = _two_blocks()
    scene.tool_settings.use_snap_anim = False
    assert _drag(obj, "right", 40) == [(0, 40), (40, 88)], "the touching block moves along"
    assert _drag(obj, "right", 40, alt=True) == [(0, 40), (48, 96)], "Alt: it stays"
    assert _drag(obj, "left", 60, uid=2) == [(12, 60), (60, 96)]
