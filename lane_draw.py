"""Drawing the block lane: the blocks, their path strips, the add chip and the text being typed."""

import math
import traceback

import blf
import bpy
import gpu
from gpu_extras.batch import batch_for_shader

from . import assist, block_text, blocks, engine, lane, return_to_start
from .lane import BASE, FONT, PAD, RADIUS, STRIP, frame_x, rect, ui_scale

SCRUB = 26  # the frame numbers along the top
ROW = 17  # word list row height
DIM = 0.55  # alpha of dimmed text: the placeholder and the ghost
AMBER = (0.95, 0.66, 0.2)
WHITE = (1, 1, 1, 0.9)

_handle = None


def _rounded(x0, y0, x1, y1, r, seg=5):
    """The outline of a rounded rectangle."""
    r = max(0.0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    corners = ((x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180), (x1 - r, y0 + r, 270))
    points = []
    for cx, cy, start in corners:
        for k in range(seg + 1):
            a = math.radians(start + 90 * k / seg)
            points.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return points


def _fill(points, color):
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    centre = (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))
    shader.uniform_float("color", color)
    batch_for_shader(shader, "TRI_FAN", {"pos": [centre, *points, points[0]]}).draw(shader)


def _line(points, color, width, closed=True):
    shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
    shader.uniform_float("lineWidth", width)
    shader.uniform_float("color", color)
    batch_for_shader(shader, "LINE_STRIP", {"pos": [*points, points[0]] if closed else points}).draw(shader)


def _text(x, y, text, color, width):
    """Text cut short with an ellipsis to fit width."""
    while text and blf.dimensions(0, text)[0] > width:
        text = text[:-2] + "…" if len(text) > 1 else ""
    blf.color(0, *color)
    blf.position(0, x, y, 0)
    blf.draw(0, text)


def draw():
    if lane._s.failed:
        return
    try:
        _draw(bpy.context)
    except Exception:  # report once: a draw callback runs on every redraw
        lane._s.failed = True
        print("SprongMotion: the block lane stopped drawing")
        traceback.print_exc()


def _draw(context):
    obj = blocks.armature(context)
    if obj is None:
        return
    s = blocks.settings(obj)
    has_path = s is not None and s.path
    region, scale, theme = context.region, ui_scale(), context.preferences.themes[0]
    y0, y1 = rect()
    gpu.state.blend_set("ALPHA")
    back = (*[c * 0.75 for c in theme.dopesheet_editor.space.back], 0.85)
    _fill(_rounded(0, y0 - (STRIP + 6 if has_path else 3) * scale, region.width, y1 + 3 * scale, 0), back)
    blf.size(0, FONT * scale)
    blf.enable(0, blf.SHADOW)
    blf.shadow(0, 3, 0, 0, 0, 0.55)
    blf.shadow_offset(0, 1, -1)
    hover = lane._s.hover[1:] if lane._s.hover[0] == region.as_pointer() else (None, -1)
    try:
        for i, b in enumerate(s.blocks if s else ()):
            hovered = hover[0] in {"left", "right", "body"} and hover[1] == i
            _draw_block(region, obj, i, b, hovered, theme)
            if has_path or return_to_start.walks_back(s, b):
                _draw_strip(region, b, hover == ("strip", i), theme)
        _draw_add_chip(region, s, hover[0] == "add", theme)
    finally:
        blf.disable(0, blf.SHADOW)
        gpu.state.blend_set("NONE")


def _draw_add_chip(region, s, hovered, theme):
    y0, y1 = rect()
    x0, x1, label = lane.add_chip(region, s)
    ink, scale = theme.dopesheet_editor.space.text, ui_scale()
    _line(_rounded(x0, y0, x1, y1, RADIUS * scale), (*ink, 0.7 if hovered else 0.35), scale)
    _text(x0 + PAD * scale, y0 + BASE * scale, label, (*ink, 1.0 if hovered else 0.7), x1 - x0)


def _draw_block(region, obj, i, b, hovered, theme):
    scale, (y0, y1) = ui_scale(), rect()
    x0, x1 = frame_x(region, b.start) + scale, frame_x(region, b.end) - scale
    if x1 < 0 or x0 > region.width:
        return
    brightness = 1.18 if hovered else 1.0
    color = [min(1.0, c * brightness) for c in b.color] if b.enabled else [0.45] * 3
    shape = _rounded(x0, y0, x1, y1, RADIUS * scale)
    has_take = b.variants and not (b.dirty and not b.locked)
    _fill(shape, (*color, 0.95 if has_take else 0.5))
    if i == b.id_data.sprongmotion.active:
        _line(shape, (*theme.view_3d.object_active, 1.0), 2 * scale)
    _draw_progress(obj, b, x0, x1, y0)
    _draw_joins(region, b, x0, y0, x1, y1)
    if block_text._state.typing and block_text._state.typing[0] == b.uid:
        _draw_typing(region, x0, y0, x1, y1, *block_text._state.typing[1:], theme)
    else:
        _draw_label(b, x0, y0, x1, y1)


def _draw_progress(obj, b, x0, x1, y0):
    run = engine.progress()
    if run and run.rig == obj.session_uid and run.uid == b.uid:
        scale = ui_scale()
        done = (x1 - x0 - 6 * scale) * run.done / run.total
        _fill(_rounded(x0 + 3 * scale, y0 + 2 * scale, x0 + 3 * scale + done, y0 + 4 * scale, scale), (1, 1, 1, 0.9))


def _draw_label(b, x0, y0, x1, y1):
    """The block's text, with its lock and "take n/m" on the right when they fit."""
    scale = ui_scale()
    left, right = max(x0, 0) + PAD * scale, x1 - PAD * scale
    if b.locked and right - left > 30 * scale:
        right -= 9 * scale
        _lock(right, (y0 + y1) / 2 - 5 * scale, scale)
        right -= 5 * scale
    tag = f"{b.variant + 1}/{b.variants}" if b.variants > 1 else ""
    if tag and right - left > 60 * scale:
        width = blf.dimensions(0, tag)[0]
        _text(right - width, y0 + BASE * scale, tag, (1, 1, 1, 0.75), width + 1)
        right -= width + 4 * scale
    if b.text.strip():
        _text(left, y0 + BASE * scale, b.text, (1, 1, 1, 1), right - left)
    else:
        _text(left, y0 + BASE * scale, "Type a motion", (1, 1, 1, DIM), right - left)


def _draw_joins(region, b, x0, y0, x1, y1):
    """Notches where the take joins the rig's own keys (amber when far off), and the range picked to regenerate."""
    scale = ui_scale()
    before, after = b.anchor_key.split("|")
    mid, h = (y0 + y1) / 2, 4 * scale
    if before:
        _fill([(x0, mid - h), (x0 + h, mid), (x0, mid + h)], WHITE)
    if after:
        _fill([(x1, mid - h), (x1 - h, mid), (x1, mid + h)], (*AMBER, 1) if b.seam_note else WHITE)
    if lane._s.range and lane._s.range[0] == b.uid:
        lo, hi = sorted(frame_x(region, f) for f in lane._s.range[1:])
        _fill(_rounded(max(lo, x0), y0, min(hi, x1), y1, 0), (1, 1, 1, 0.3))


def _draw_strip(region, b, hovered, theme):
    """The path strip: grey where the path fits the motion, amber where it asks more, with the reason on hover."""
    scale, (y0, y1) = ui_scale(), rect()
    x0, x1 = frame_x(region, b.start) + scale, frame_x(region, b.end) - scale
    if x1 < 0 or x0 > region.width:
        return
    grey = (*theme.dopesheet_editor.space.text, 0.25)
    color = (*AMBER, 0.95 if hovered else 0.8) if b.path_note else grey
    _fill(_rounded(x0, y0 - (STRIP + 2) * scale, x1, y0 - 2 * scale, STRIP * scale / 2), color)
    if hovered and b.path_note:
        width = blf.dimensions(0, b.path_note)[0] + 2 * PAD * scale
        left = max(0, min(x0, region.width - width))
        _fill(_rounded(left, y1 + 4 * scale, left + width, y1 + 22 * scale, RADIUS * scale), (0, 0, 0, 0.7))
        _text(left + PAD * scale, y1 + 10 * scale, b.path_note, (*AMBER, 1), width)


def _lock(x, y, scale, color=WHITE):
    _fill(_rounded(x, y, x + 9 * scale, y + 6.5 * scale, 1.5 * scale), color)
    shackle = []
    for k in range(11):
        a = math.pi * k / 10
        shackle.append((x + 4.5 * scale + 2.8 * scale * math.cos(a), y + 6.5 * scale + 3.3 * scale * math.sin(a)))
    _line(shackle, color, 1.6 * scale, closed=False)


def _draw_typing(region, x0, y0, x1, y1, text, caret, theme):
    """The text field while typing in a block: text, ghost, caret and the word list."""
    scale, help_ = ui_scale(), block_text._state.help
    field = _rounded(x0 + 2 * scale, y0 + 2 * scale, x1 - 2 * scale, y1 - 2 * scale, (RADIUS - 1) * scale)
    _fill(field, (0, 0, 0, 0.45))
    _line(field, (1, 1, 1, 0.9), 1.5 * scale)
    left, width = x0 + PAD * scale, x1 - x0 - 2 * PAD * scale
    before_caret = blf.dimensions(0, text[:caret])[0]
    scroll = max(0.0, before_caret - width)
    blf.clipping(0, left, y0, left + width, y1)
    blf.enable(0, blf.CLIPPING)
    for at, part, alpha in ((0, text, 1), (blf.dimensions(0, text)[0], help_.ghost, DIM)):
        blf.color(0, 1, 1, 1, alpha)
        blf.position(0, left - scroll + at, y0 + BASE * scale, 0)
        blf.draw(0, part)
    blf.disable(0, blf.CLIPPING)
    x = left + before_caret - scroll
    _line([(x, y0 + 5 * scale), (x, y1 - 5 * scale)], (1, 1, 1, 1), 1.5 * scale, closed=False)
    if help_.rows:
        word_start = assist.insert(text, caret, "")[1]
        word_x = left - scroll + blf.dimensions(0, text[:word_start])[0] - PAD * scale
        _draw_rows(region, word_x, y1, help_.rows, help_.pick, theme)


def _draw_rows(region, x, y, rows, pick, theme):
    """The word list above the block, below the frame numbers: as many rows as fit."""
    scale, ui = ui_scale(), theme.user_interface
    n = min(len(rows), int((region.height - SCRUB * scale - y - 8 * scale) // (ROW * scale)))
    if n < 1:
        return
    width = max(blf.dimensions(0, r)[0] for r in rows[:n]) + 2 * PAD * scale
    x = max(0.0, min(x, region.width - width))
    y0 = y + 4 * scale
    _fill(_rounded(x, y0, x + width, y0 + n * ROW * scale + 4 * scale, RADIUS * scale), ui.wcol_menu_back.inner)
    for k, row in enumerate(rows[:n]):
        row_y = y0 + 2 * scale + (n - 1 - k) * ROW * scale
        if k == pick:
            highlight = _rounded(x + 2 * scale, row_y, x + width - 2 * scale, row_y + ROW * scale, (RADIUS - 2) * scale)
            _fill(highlight, ui.wcol_menu_item.inner_sel)
        ink = ui.wcol_menu_item.text_sel if k == pick else ui.wcol_menu_back.text
        _text(x + PAD * scale, row_y + 5 * scale, row, (*ink, 1), width)


def register():
    global _handle
    _handle = bpy.types.SpaceDopeSheetEditor.draw_handler_add(draw, (), "WINDOW", "POST_PIXEL")


def unregister():
    global _handle
    if _handle:
        bpy.types.SpaceDopeSheetEditor.draw_handler_remove(_handle, "WINDOW")
    _handle = None
