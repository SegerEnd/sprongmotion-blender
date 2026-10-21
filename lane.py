"""The block lane on the Timeline / Dope Sheet: drawing (POST_PIXEL, gpu + blf), mouse operators, text editing."""

import math
import time
import traceback

import blf
import bpy
import gpu
from bpy.props import BoolProperty, IntProperty, StringProperty
from gpu_extras.batch import batch_for_shader

from . import assist, blocks, engine, ops, paths

HEIGHT, BOTTOM, EDGE, RADIUS, PAD = 34, 40, 6, 5, 8  # pixels at UI scale 1; BOTTOM clears the scrollbar and markers
FONT, BASE = 12, 12  # text size, its baseline over the block's bottom
SCRUB = 26  # the frame numbers along the top
DIM = 0.55  # the alpha of dimmed text in the lane (the placeholder, the ghost)
STRIP = 3  # the path strip under the blocks
AMBER = (0.95, 0.66, 0.2)
DRAG_START = 4
SNAP = 8  # pixels from a snap target that snap to it
PAUSE = 0.3  # seconds without a key before the word list shows
ROW = 17  # word list row height


class _State:
    def __init__(self):
        self.handle, self.failed = None, False
        self.hover = (0, None, -1)  # region pointer, hit kind, block index
        self.typing = None  # [block uid, text, caret] while a block's text is typed in place
        self.range = None  # (block uid, frame, frame) while Alt+dragging a range to regenerate
        self.help = _Help()
        self.menu_frame = self.menu_gap = 0


class _Help:
    """What the typing help shows for the text being typed: ghost after the caret, or rows (the word list) with pick
    highlighted. quiet: the text Esc hid them for, nothing shows until it changes."""

    def __init__(self, text=None):
        self.sentence, self.ghost, self.rows, self.pick, self.quiet, self.last = "", "", [], 0, text, 0.0


_s = _State()


def _u():
    return bpy.context.preferences.system.ui_scale or 1.0  # 0 without a window


def rect():
    return BOTTOM * _u(), (BOTTOM + HEIGHT) * _u()


def frame_x(region, frame):
    return region.view2d.view_to_region(frame, 0, clip=False)[0]


def x_frame(region, x):
    return region.view2d.region_to_view(x, 0)[0]


def snapped(scene, at, reach, edges=()):
    """How far to shift the frames at (floats) to put the nearest one on a snap target within reach (frames): the
    given block edges, the playhead, and Blender's Snap To element (Frame, Second or Nearest Marker). None if none
    is in reach."""
    targets, kind = [*edges, scene.frame_current], scene.tool_settings.snap_anim_element
    if kind == "MARKER":
        targets += [m.frame for m in scene.timeline_markers]
    best = None
    for f in at:
        near = [round(round(f / blocks.fps(scene)) * blocks.fps(scene))] if kind == "SECOND" else []
        for g in targets + near:
            if abs(g - f) <= reach and (best is None or abs(g - f) < abs(best)):
                best = g - f
    return best


def add_chip(region, s):
    """Right after the last block, or at the playhead on an empty lane."""
    u = _u()
    blf.size(0, FONT * u)
    if s is not None and len(s.blocks):
        x0, label = frame_x(region, s.blocks[-1].end) + 2 * u, "+"
    else:
        x0, label = frame_x(region, bpy.context.scene.frame_current), "+ Add motion"
    return x0, x0 + blf.dimensions(0, label)[0] + 2 * PAD * u, label


def hit(region, s, x, y):
    y0, y1 = rect()
    strips = s is not None and (s.path or any(paths.walks_back(s, b) for b in s.blocks))
    if strips and y0 - (STRIP + 4) * _u() <= y < y0:
        return next((("strip", i) for i, b in enumerate(s.blocks)
                     if frame_x(region, b.start) <= x < frame_x(region, b.end)), (None, -1))  # fmt: skip
    if not y0 <= y <= y1:
        return None, -1
    x0, x1, _ = add_chip(region, s)
    if x0 <= x <= x1:
        return "add", -1
    xs, e = [(frame_x(region, b.start), frame_x(region, b.end)) for b in (s.blocks if s else ())], EDGE * _u()
    for i, (a, b) in enumerate(xs):  # where two blocks touch, the edge is the left one's end
        if abs(x - b) <= e:
            return "right", i
        if abs(x - a) <= e and (i == 0 or a - xs[i - 1][1] > e):
            return "left", i
    for i, (a, b) in enumerate(xs):
        if a < x < b:
            return "body", i
        if i and xs[i - 1][1] < x < a:
            return "gap", i - 1
    return None, -1


def _rounded(x0, y0, x1, y1, r, seg=5):
    r = max(0.0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    corners = ((x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180), (x1 - r, y0 + r, 270))
    return [
        (cx + r * math.cos(math.radians(a0 + 90 * k / seg)), cy + r * math.sin(math.radians(a0 + 90 * k / seg)))
        for cx, cy, a0 in corners
        for k in range(seg + 1)
    ]


def _fill(pts, color):
    sh = gpu.shader.from_builtin("UNIFORM_COLOR")
    centre = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    sh.uniform_float("color", color)
    batch_for_shader(sh, "TRI_FAN", {"pos": [centre, *pts, pts[0]]}).draw(sh)


def _line(pts, color, width, closed=True):
    sh = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    sh.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
    sh.uniform_float("lineWidth", width)
    sh.uniform_float("color", color)
    batch_for_shader(sh, "LINE_STRIP", {"pos": [*pts, pts[0]] if closed else pts}).draw(sh)


def _text(x, y, text, color, width):
    while text and blf.dimensions(0, text)[0] > width:
        text = text[:-2] + "…" if len(text) > 1 else ""
    blf.color(0, *color)
    blf.position(0, x, y, 0)
    blf.draw(0, text)


def draw():
    if _s.failed:
        return
    try:
        _draw(bpy.context)
    except Exception:  # report once: a draw callback runs on every redraw
        _s.failed = True
        print("SprongMotion: the block lane stopped drawing")
        traceback.print_exc()


def _draw(context):
    obj = blocks.armature(context)
    if obj is None:
        return
    s = blocks.settings(obj)
    path = s is not None and s.path
    region, u, theme = context.region, _u(), context.preferences.themes[0]
    y0, y1 = rect()
    gpu.state.blend_set("ALPHA")
    _fill(
        _rounded(0, y0 - (STRIP + 6 if path else 3) * u, region.width, y1 + 3 * u, 0),
        (*[c * 0.75 for c in theme.dopesheet_editor.space.back], 0.85),
    )
    blf.size(0, FONT * u)
    blf.enable(0, blf.SHADOW)
    blf.shadow(0, 3, 0, 0, 0, 0.55)
    blf.shadow_offset(0, 1, -1)
    hover = _s.hover[1:] if _s.hover[0] == region.as_pointer() else (None, -1)
    try:
        for i, b in enumerate(s.blocks if s else ()):
            _draw_block(region, obj, i, b, hover[0] in {"left", "right", "body"} and hover[1] == i, theme)
            if path or paths.walks_back(s, b):
                _draw_strip(region, b, hover == ("strip", i), theme)
        x0, x1, label = add_chip(region, s)
        ink = theme.dopesheet_editor.space.text
        _line(_rounded(x0, y0, x1, y1, RADIUS * u), (*ink, 0.7 if hover[0] == "add" else 0.35), u)
        _text(x0 + PAD * u, y0 + BASE * u, label, (*ink, 1.0 if hover[0] == "add" else 0.7), x1 - x0)
    finally:
        blf.disable(0, blf.SHADOW)
        gpu.state.blend_set("NONE")


def _draw_block(region, obj, i, b, hovered, theme):
    u, (y0, y1) = _u(), rect()
    x0, x1 = frame_x(region, b.start) + u, frame_x(region, b.end) - u
    if x1 < 0 or x0 > region.width:
        return
    color = [min(1.0, c * (1.18 if hovered else 1.0)) for c in b.color] if b.enabled else [0.45] * 3
    shape = _rounded(x0, y0, x1, y1, RADIUS * u)
    _fill(shape, (*color, 0.95 if b.variants and not (b.dirty and not b.locked) else 0.5))
    if i == b.id_data.sprongmotion.active:
        _line(shape, (*theme.view_3d.object_active, 1.0), 2 * u)
    run = engine.progress()
    if run and run.rig == obj.session_uid and run.uid == b.uid:
        _fill(
            _rounded(x0 + 3 * u, y0 + 2 * u, x0 + 3 * u + (x1 - x0 - 6 * u) * run.done / run.total, y0 + 4 * u, u),
            (1, 1, 1, 0.9),
        )
    _draw_joins(region, b, x0, y0, x1, y1)
    if _s.typing and _s.typing[0] == b.uid:
        return _draw_typing(region, x0, y0, x1, y1, *_s.typing[1:], theme)
    tag = f"{b.variant + 1}/{b.variants}" if b.variants > 1 else ""
    left, right = max(x0, 0) + PAD * u, x1 - PAD * u
    if b.locked and right - left > 30 * u:
        right -= 9 * u
        _lock(right, (y0 + y1) / 2 - 5 * u, u)
        right -= 5 * u
    if tag and right - left > 60 * u:
        w = blf.dimensions(0, tag)[0]
        _text(right - w, y0 + BASE * u, tag, (1, 1, 1, 0.75), w + 1)
        right -= w + 4 * u
    text, white = (b.text, (1, 1, 1, 1)) if b.text.strip() else ("Type a motion", (1, 1, 1, DIM))
    _text(left, y0 + BASE * u, text, white, right - left)


def _draw_joins(region, b, x0, y0, x1, y1):
    """A notch on the edge where the take joins the rig's own keys (amber at the far end when it is far off), and
    the range being picked to regenerate."""
    u, before, after = _u(), *b.anchor_key.split("|")
    mid, h = (y0 + y1) / 2, 4 * u
    if before:
        _fill([(x0, mid - h), (x0 + h, mid), (x0, mid + h)], (1, 1, 1, 0.9))
    if after:
        _fill([(x1, mid - h), (x1 - h, mid), (x1, mid + h)], (*AMBER, 1) if b.seam_note else (1, 1, 1, 0.9))
    if _s.range and _s.range[0] == b.uid:
        lo, hi = sorted(frame_x(region, f) for f in _s.range[1:])
        _fill(_rounded(max(lo, x0), y0, min(hi, x1), y1, 0), (1, 1, 1, 0.3))


def _draw_strip(region, b, hovered, theme):
    """Grey where the path fits the block's motion, amber where it asks more; hovered, the reason above the block."""
    u, (y0, _) = _u(), rect()
    x0, x1 = frame_x(region, b.start) + u, frame_x(region, b.end) - u
    if x1 < 0 or x0 > region.width:
        return
    ink = theme.dopesheet_editor.space.text
    color = (*AMBER, 0.95 if hovered else 0.8) if b.path_note else (*ink, 0.25)
    _fill(_rounded(x0, y0 - (STRIP + 2) * u, x1, y0 - 2 * u, STRIP * u / 2), color)
    if hovered and b.path_note:
        w = blf.dimensions(0, b.path_note)[0] + 2 * PAD * u
        bx = max(0, min(x0, region.width - w))
        _, y1 = rect()
        _fill(_rounded(bx, y1 + 4 * u, bx + w, y1 + 22 * u, RADIUS * u), (0, 0, 0, 0.7))
        _text(bx + PAD * u, y1 + 10 * u, b.path_note, (*AMBER, 1), w)


def _lock(x, y, u, color=(1, 1, 1, 0.9)):
    _fill(_rounded(x, y, x + 9 * u, y + 6.5 * u, 1.5 * u), color)
    shackle = [(x + 4.5 * u + 2.8 * u * math.cos(a), y + 6.5 * u + 3.3 * u * math.sin(a))
               for a in (math.pi * k / 10 for k in range(11))]  # fmt: skip
    _line(shackle, color, 1.6 * u, closed=False)


def _draw_typing(region, x0, y0, x1, y1, text, caret, theme):
    u, h = _u(), _s.help
    field = _rounded(x0 + 2 * u, y0 + 2 * u, x1 - 2 * u, y1 - 2 * u, (RADIUS - 1) * u)
    _fill(field, (0, 0, 0, 0.45))
    _line(field, (1, 1, 1, 0.9), 1.5 * u)
    left, width = x0 + PAD * u, x1 - x0 - 2 * PAD * u
    pre = blf.dimensions(0, text[:caret])[0]
    shift = max(0.0, pre - width)
    blf.clipping(0, left, y0, left + width, y1)
    blf.enable(0, blf.CLIPPING)
    for at, part, alpha in ((0, text, 1), (blf.dimensions(0, text)[0], h.ghost, DIM)):
        blf.color(0, 1, 1, 1, alpha)
        blf.position(0, left - shift + at, y0 + BASE * u, 0)
        blf.draw(0, part)
    blf.disable(0, blf.CLIPPING)
    x = left + pre - shift
    _line([(x, y0 + 5 * u), (x, y1 - 5 * u)], (1, 1, 1, 1), 1.5 * u, closed=False)
    if h.rows:
        word = assist.insert(text, caret, "")[1]
        _draw_rows(region, left - shift + blf.dimensions(0, text[:word])[0] - PAD * u, y1, h.rows, h.pick, theme)


def _draw_rows(region, x, y, rows, pick, theme):
    """The word list above the block (below it are the markers and the scrollbar), clear of the frame numbers at the
    top; as many rows as fit."""
    u, ui = _u(), theme.user_interface
    n = min(len(rows), int((region.height - SCRUB * u - y - 8 * u) // (ROW * u)))
    if n < 1:
        return
    w = max(blf.dimensions(0, r)[0] for r in rows[:n]) + 2 * PAD * u
    x = max(0.0, min(x, region.width - w))
    y0 = y + 4 * u
    _fill(_rounded(x, y0, x + w, y0 + n * ROW * u + 4 * u, RADIUS * u), ui.wcol_menu_back.inner)
    for k, r in enumerate(rows[:n]):
        ry = y0 + 2 * u + (n - 1 - k) * ROW * u
        if k == pick:
            _fill(_rounded(x + 2 * u, ry, x + w - 2 * u, ry + ROW * u, (RADIUS - 2) * u), ui.wcol_menu_item.inner_sel)
        ink = ui.wcol_menu_item.text_sel if k == pick else ui.wcol_menu_back.text
        _text(x + PAD * u, ry + 5 * u, r, (*ink, 1), w)


def type_key(text, caret, event, clipboard=""):
    """One key press on a single-line text field: (text, caret) after it."""
    t, cmd = event.type, event.ctrl or event.oskey
    moves = {"LEFT_ARROW": caret - 1, "RIGHT_ARROW": caret + 1, "HOME": 0, "END": len(text)}
    if t in moves:
        return text, max(0, min(len(text), moves[t]))
    if t == "BACK_SPACE" and caret:
        return text[: caret - 1] + text[caret:], caret - 1
    if t == "DEL":
        return text[:caret] + text[caret + 1 :], caret
    insert = (clipboard.splitlines() or [""])[0] if t == "V" and cmd else event.unicode if not cmd else ""
    if insert and insert.isprintable():
        return text[:caret] + insert + text[caret:], caret + len(insert)
    return text, caret


def begin(uid, text):
    """Start typing in a block: nothing shows until the text changes."""
    _s.typing, _s.help = [uid, text, len(text)], _Help(text)


def typed(event, clipboard=""):
    """One key while typing in a block, not one that ends the edit: the typing help's, else the text field's."""
    h, (_, text, caret) = _s.help, _s.typing
    took = assist.take(assist.kind(event), text, caret, h.ghost, h.rows, h.pick)
    if took is None:
        took = (*type_key(text, caret, event, clipboard), 0, False)
    text2, caret2, h.pick, hide = took
    if hide:
        h.quiet, h.ghost, h.rows = text2, "", []
    elif (text2, caret2) != (text, caret):
        _s.typing[1:] = text2, caret2
        changed()


def help_key(event):
    """True when the typing help takes this key (Enter or Esc while it shows something)."""
    h, (_, text, caret) = _s.help, _s.typing
    return assist.take(assist.kind(event), text, caret, h.ghost, h.rows, h.pick) is not None


def changed():
    """After the text or caret moved: the ghost now (cheap: the shown sentence is kept while it fits, else one
    cached encode), the list after a pause."""
    h, (_, text, caret) = _s.help, _s.typing
    helper = engine.helper()
    h.rows, h.pick, h.last = [], 0, time.perf_counter()
    on = helper is not None and text != h.quiet
    h.sentence = helper.sentence(text, h.sentence) if on and caret == len(text) else ""
    h.ghost = h.sentence[len(text) :]
    if on and not bpy.app.timers.is_registered(_pause):
        bpy.app.timers.register(_pause, first_interval=PAUSE)


def _pause():
    h, helper = _s.help, engine.helper()
    if _s.typing is None or helper is None:
        return None
    wait = PAUSE - (time.perf_counter() - h.last)
    if wait > 0:
        return wait
    _, text, caret = _s.typing
    h.rows = [] if h.ghost or text == h.quiet else helper.complete(text, caret)
    if h.rows:
        blocks.redraw()
    return None


def _in_lane(context):
    return bool(context.area and context.area.type == "DOPESHEET_EDITOR" and context.region.type == "WINDOW")


class _LaneOp:
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return _in_lane(context) and ops.editable(context)  # read-only blocks: the lane only shows them

    def hit(self, context, event):
        return hit(context.region, blocks.settings(context.active_object), event.mouse_region_x, event.mouse_region_y)


def _suggest(_op, _context, text):
    """The popup's list: a native text field cannot draw ghost text, so the sentence the ghost would show comes
    first, then the word completions."""
    helper = engine.helper()
    if helper is None:
        return []
    ghost, words = helper.ghost(text), helper.complete(text, len(text))
    return (([text + ghost] if ghost else []) + [assist.insert(text, len(text), w)[0] for w in words])[: assist.ROWS]


class SM_OT_block_edit(bpy.types.Operator):
    """Edit the block's text: in the block itself or in a popup (the popup also takes input methods)"""

    bl_idname = "sm.block_edit"
    bl_label = "Edit Block Text"
    bl_options = {"REGISTER", "UNDO"}
    index: IntProperty(default=-1, options={"SKIP_SAVE"})
    popup: BoolProperty(options={"SKIP_SAVE"})
    text: StringProperty(name="Text", search=_suggest)

    poll = classmethod(lambda cls, context: blocks.has_blocks(context) and ops.editable(context))

    def block(self, context):
        obj, i = blocks.pick(context, self.index)
        return blocks.settings(obj).blocks[i]

    def invoke(self, context, _event):
        b = self.block(context)
        self.text = b.text
        p = engine.prefs()
        if self.popup or not _in_lane(context) or (p and p.text_edit == "POPUP"):
            return context.window_manager.invoke_props_dialog(self, width=420, title="Motion", confirm_text="Apply")
        begin(b.uid, b.text)
        self.region = context.region
        context.window_manager.modal_handler_add(self)
        self.region.tag_redraw()
        return {"RUNNING_MODAL"}

    def draw(self, _context):
        self.layout.activate_init = True
        self.layout.prop(self, "text", text="")

    def modal(self, context, event):
        if event.value != "PRESS":
            return {"RUNNING_MODAL"}
        if event.type in {"RET", "NUMPAD_ENTER", "ESC", "LEFTMOUSE", "RIGHTMOUSE"} and not help_key(event):
            self.text, _s.typing = _s.typing[1], None
            self.region.tag_redraw()
            if event.type == "ESC":
                return {"CANCELLED"}
            done = self.execute(context)
            return done | {"PASS_THROUGH"} if "MOUSE" in event.type else done
        typed(event, context.window_manager.clipboard)
        self.region.tag_redraw()
        return {"RUNNING_MODAL"}

    def execute(self, context):
        b = self.block(context)
        if b.text == self.text:
            return {"CANCELLED"}
        b.text = self.text
        return {"FINISHED"}


class SM_OT_lane_hover(_LaneOp, bpy.types.Operator):
    bl_idname = "sm.lane_hover"
    bl_label = "Block Lane Hover"
    poll = classmethod(lambda cls, context: _in_lane(context) and blocks.armature(context) is not None)

    def invoke(self, context, event):
        state = (context.region.as_pointer(), *self.hit(context, event))
        if state != _s.hover:
            _s.hover = state
            context.region.tag_redraw()
        return {"PASS_THROUGH"}


class SM_OT_lane_click(_LaneOp, bpy.types.Operator):
    """Select a block; drag its middle to move it (Shift: move all, Alt: pick a range to regenerate), drag an edge
    to resize, the blocks touching it moving along (Alt: they stay, Ctrl: snap the other way)"""

    bl_idname = "sm.lane_click"
    bl_label = "Edit Blocks"
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}

    def invoke(self, context, event):
        kind, i = self.hit(context, event)
        obj = context.active_object
        if kind in {None, "strip", "gap"}:
            return {"PASS_THROUGH"}
        if kind == "add":
            text = obj.sprongmotion.prompt
            blocks.add(obj, text)
            if not text.strip():
                bpy.ops.sm.block_edit("INVOKE_DEFAULT", index=len(blocks.settings(obj).blocks) - 1)
            return {"FINISHED"}
        s = blocks.settings(obj)
        s.active = i
        kind = "range" if kind == "body" and event.alt else kind
        self.kind, self.uid, self.shift, self.moved = kind, s.blocks[i].uid, event.shift, False
        self.before = blocks.continues(s)
        self.x0, self.f0 = event.mouse_region_x, x_frame(context.region, event.mouse_region_x)
        self.snapshot, self.region = (
            [(b.uid, b.start, b.end, b.locked, b.dirty, b.remake) for b in s.blocks],
            context.region,
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        obj = context.active_object
        if event.type == "MOUSEMOVE":
            self.moved |= abs(event.mouse_region_x - self.x0) >= DRAG_START * _u()
            if self.moved:
                self.drag(obj, event.mouse_region_x, event.ctrl, event.alt)
                self.region.tag_redraw()
        elif event.type == "LEFTMOUSE" and event.value == "RELEASE":
            if not self.moved:
                return {"CANCELLED"}
            self.done(obj)
            return {"FINISHED"}
        elif event.type in {"ESC", "RIGHTMOUSE"}:
            self.restore(obj)
            self.region.tag_redraw()
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def drag(self, obj, x, ctrl=False, alt=False):
        """From the blocks as they were at the click: an edge to frame, or a block (Shift: all) moved by the drag."""
        s, scene, f = blocks.settings(obj), bpy.context.scene, x_frame(self.region, x)
        _put(s, self.snapshot)
        if self.kind == "range":
            _s.range = (self.uid, round(self.f0), round(f))
            return
        i, spans = blocks.find(s, self.uid), [(uid, a, b) for uid, a, b, *_ in self.snapshot]
        if self.kind == "body":
            at = [e + f - self.f0 for uid, a, b in spans if self.shift or uid == self.uid for e in (a, b)]
            targets = [] if self.shift else [e for uid, a, b in spans if uid != self.uid for e in (a, b)]
        else:
            own = spans[i][2 if self.kind == "right" else 1]
            at, targets = [f], [e for _, a, b in spans for e in (a, b) if e != own]
        shift = None
        if scene.tool_settings.use_snap_anim != ctrl:
            shift = snapped(scene, at, SNAP * _u() / (frame_x(self.region, 1) - frame_x(self.region, 0)), targets)
        frame, moved = (
            (round(f), round(f) - round(self.f0)) if shift is None else (round(f + shift), round(f - self.f0 + shift))
        )
        if self.kind == "right":
            blocks.resize(obj, i, end=frame, follow=not alt)  # Alt: the others stay
        elif self.kind == "left":
            blocks.resize(obj, i, start=frame, follow=not alt)
        elif self.shift:
            blocks.shift(s, range(len(s.blocks)), moved)
        else:
            s.active = blocks.place(s, i, s.blocks[i].start + moved)

    def done(self, obj):
        s = blocks.settings(obj)
        if self.kind == "range":
            picked, _s.range = _s.range, None
            why = engine.remake_range(obj, blocks.find(s, self.uid), *sorted(picked[1:]))
            if why:
                self.report({"WARNING"}, why)
        elif self.shift and self.kind == "body":
            if len(s.blocks) == 1:
                blocks.snap(obj, 0, (s.blocks[0].start + s.blocks[0].end) // 2)
            engine.edited(obj)
        elif self.kind == "body":  # a block that now goes on from another one (or none) is made again
            blocks.mark_dirty(obj, *(u for u, p in blocks.continues(s).items() if self.before.get(u) != p))
        else:
            blocks.mark_dirty(obj, self.uid)

    def restore(self, obj):
        s, _s.range = blocks.settings(obj), None
        _put(s, self.snapshot)
        for b, (_, _, _, locked, dirty, remake) in zip(s.blocks, self.snapshot, strict=True):
            b.dirty, b.remake = dirty, remake
            if b.locked != locked:
                b["locked"] = locked  # an ID write: the RNA update would refit


def _put(s, snapshot):
    """The blocks' order and places as in snapshot [(uid, start, end, ...)]."""
    for k, (uid, start, end, *_) in enumerate(snapshot):
        s.blocks.move(blocks.find(s, uid), k)
        s.blocks[k].start, s.blocks[k].end = start, end


class SM_OT_lane_open(_LaneOp, bpy.types.Operator):
    bl_idname = "sm.lane_open"
    bl_label = "Edit Block Text"

    def invoke(self, context, event):
        kind, i = self.hit(context, event)
        if kind != "body":
            return {"PASS_THROUGH"}
        bpy.ops.sm.block_edit("INVOKE_DEFAULT", index=i)
        return {"FINISHED"}


class SM_OT_lane_menu(_LaneOp, bpy.types.Operator):
    bl_idname = "sm.lane_menu"
    bl_label = "Block Menu"

    def invoke(self, context, event):
        kind, i = self.hit(context, event)
        if kind not in {"body", "gap"}:
            return {"PASS_THROUGH"}
        if kind == "body":
            blocks.settings(context.active_object).active = i
        _s.menu_frame, _s.menu_gap = round(x_frame(context.region, event.mouse_region_x)), i
        bpy.ops.wm.call_menu(name=(SM_MT_block if kind == "body" else SM_MT_gap).__name__)
        return {"FINISHED"}


class SM_OT_lane_delete(_LaneOp, bpy.types.Operator):
    bl_idname = "sm.lane_delete"
    bl_label = "Delete Block"
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}

    def invoke(self, context, event):
        kind, i = self.hit(context, event)
        if kind in {None, "add", "strip", "gap"}:
            return {"PASS_THROUGH"}
        blocks.remove(context.active_object, i)
        return {"FINISHED"}


class SM_OT_lane_key(_LaneOp, bpy.types.Operator):
    """Copy, paste or duplicate the selected block with the mouse over the lane"""

    bl_idname = "sm.lane_key"
    bl_label = "Block Key"
    action: StringProperty()

    def invoke(self, context, event):
        y0, y1 = rect()
        if not y0 <= event.mouse_region_y <= y1 or (self.action != "paste" and not blocks.has_blocks(context)):
            return {"PASS_THROUGH"}
        return getattr(bpy.ops.sm, f"block_{self.action}")()


class SM_MT_block(bpy.types.Menu):
    bl_label = "Block"

    def draw(self, context):
        s = blocks.settings(context.active_object)
        layout, b = self.layout, s.blocks[s.active]
        layout.operator("sm.block_edit", text="Edit Text", icon="GREASEPENCIL")
        layout.operator("sm.block_edit", text="Edit in Popup…").popup = True
        layout.operator("sm.block_split", text="Split Here", icon="CENTER_ONLY").frame = _s.menu_frame
        layout.operator("sm.block_copy", text="Copy", icon="COPYDOWN")
        layout.operator("sm.block_paste", text="Paste", icon="PASTEDOWN")
        layout.operator("sm.block_duplicate", text="Duplicate", icon="DUPLICATE")
        layout.separator()
        layout.operator("sm.block_regenerate", text="Regenerate", icon="FILE_REFRESH")
        layout.operator("sm.block_regenerate", text="New Seed", icon="MOD_NOISE").new_seed = True
        layout.operator("sm.block_regenerate_range", text="Regenerate Preview Range")
        layout.prop(b, "locked", text="Lock")
        layout.prop(b, "fresh")
        layout.prop(b, "in_place")
        layout.prop(b, "back_to_start")
        if b.variants > 1:
            layout.operator("sm.variant", text="Next Take", icon="TRIA_RIGHT").step = 1
            layout.operator("sm.variant", text="Previous Take", icon="TRIA_LEFT").step = -1
        layout.separator()
        layout.operator("sm.block_remove", text="Delete", icon="X")


class SM_MT_gap(bpy.types.Menu):
    bl_label = "Gap"

    def draw(self, _context):
        self.layout.operator("sm.fill_gap", icon="KEYFRAME").after = _s.menu_gap


CLASSES = (SM_OT_block_edit, SM_OT_lane_hover, SM_OT_lane_click, SM_OT_lane_open, SM_OT_lane_menu, SM_OT_lane_delete,
           SM_OT_lane_key, SM_MT_block, SM_MT_gap)  # fmt: skip
KEYS = [  # Dope Sheet / Timeline
    ("sm.lane_hover", "MOUSEMOVE", "ANY", {}, {}),
    ("sm.lane_click", "LEFTMOUSE", "PRESS", {}, {}),
    ("sm.lane_click", "LEFTMOUSE", "PRESS", {"shift": True}, {}),
    ("sm.lane_click", "LEFTMOUSE", "PRESS", {"ctrl": True}, {}),
    ("sm.lane_click", "LEFTMOUSE", "PRESS", {"shift": True, "ctrl": True}, {}),
    ("sm.lane_click", "LEFTMOUSE", "PRESS", {"alt": True}, {}),
    ("sm.lane_open", "LEFTMOUSE", "DOUBLE_CLICK", {}, {}),
    ("sm.lane_menu", "RIGHTMOUSE", "PRESS", {}, {}),
    ("sm.lane_delete", "X", "PRESS", {}, {}),
    ("sm.lane_delete", "DEL", "PRESS", {}, {}),
    ("sm.lane_key", "C", "PRESS", {"ctrl": True}, {"action": "copy"}),
    ("sm.lane_key", "C", "PRESS", {"oskey": True}, {"action": "copy"}),  # Cmd on macOS, as Blender's own copy
    ("sm.lane_key", "V", "PRESS", {"ctrl": True}, {"action": "paste"}),
    ("sm.lane_key", "V", "PRESS", {"oskey": True}, {"action": "paste"}),
    ("sm.lane_key", "D", "PRESS", {"shift": True}, {"action": "duplicate"}),
]


_keymaps = []


def reset():
    _s.failed, _s.typing = False, None
    if bpy.app.timers.is_registered(_pause):
        bpy.app.timers.unregister(_pause)


def register():
    global _s
    _s = _State()
    for c in CLASSES:
        bpy.utils.register_class(c)
    ops.add_keymaps([("Dopesheet", "DOPESHEET_EDITOR", KEYS)], _keymaps)
    _s.handle = bpy.types.SpaceDopeSheetEditor.draw_handler_add(draw, (), "WINDOW", "POST_PIXEL")


def unregister():
    global _s
    if _s.handle:
        bpy.types.SpaceDopeSheetEditor.draw_handler_remove(_s.handle, "WINDOW")
    reset()
    _s = _State()
    ops.remove_keymaps(_keymaps)
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
