import os
import time
from types import SimpleNamespace

import bpy
from bl_ext.smtest.sprongmotion import blocks, engine, lane


def key(type_, unicode=""):
    return SimpleNamespace(type=type_, unicode=unicode, ctrl=False, oskey=False)


def start(text=""):
    assert bpy.ops.sm.add_rig() == {"FINISHED"}, engine.error()
    obj = bpy.context.active_object
    obj.sprongmotion.auto = False
    blocks.add(obj, text, 2)
    assert engine.model() is not None, engine.error()
    lane.begin(blocks.settings(obj).blocks[0].uid, text)
    return lane._s.help


def write(text):
    for ch in text:
        lane.typed(key("SPACE" if ch == " " else ch.upper(), ch))


def press(type_):
    took = lane.help_key(key(type_))
    lane.typed(key(type_))
    return took


def paused(h):
    h.last -= lane.PAUSE
    lane._pause()


def text():
    return lane._s.typing[1]


def test_ghost_tab_takes_it():
    h = start()
    write("a person sits d")
    assert h.ghost.startswith("own on ") and h.rows == []
    paused(h)
    assert h.rows == [], "no word list while a ghost shows"
    assert press("TAB") and h.ghost == ""
    assert text().startswith("a person sits down on ") and text() in engine.helper().sentences


def test_esc_dismisses_and_typing_on_shows_again():
    h = start()
    write("a person sits d")
    assert press("ESC") and h.ghost == "" and lane._s.typing is not None, "Esc hides the ghost, the edit goes on"
    paused(h)
    assert h.rows == []
    write("o")
    assert h.ghost.startswith("wn ")
    assert not lane.help_key(key("RET")), "Enter ends the edit, it does not take the ghost"


def test_word_list_only_after_the_pause():
    h = start()
    write("the man stro")
    assert h.ghost == "" and h.rows == []
    lane._pause()
    assert h.rows == [], "no list before the pause"
    paused(h)
    assert 0 < len(h.rows) <= 5 and all(w.startswith("stro") for w in h.rows)
    pick = h.rows[1]
    assert press("DOWN_ARROW") and h.pick == 1
    assert press("RET") and text() == "the man " + pick and h.rows == []


def test_nothing_for_empty_text():
    h = start()
    paused(h)
    write(" ")
    paused(h)
    assert (h.ghost, h.rows) == ("", [])
    assert lane._suggest(None, None, "") == []


def test_preference_off_shows_nothing():
    engine.prefs().typing_help = False
    try:
        h = start()
        write("a person sits d")
        paused(h)
        write(" the man stro")
        paused(h)
        assert (h.ghost, h.rows) == ("", []) and lane._suggest(None, None, "a person sits d") == []
        assert engine.helper() is None, "the sidebar's note then leaves out the nearest words"
    finally:
        engine.prefs().typing_help = True


def test_popup_list_and_nearest_words():
    start()
    got = lane._suggest(None, None, "a person sits d")
    assert got[0] in engine.helper().sentences and got[0].startswith("a person sits down")
    got = lane._suggest(None, None, "the man stro")
    assert 0 < len(got) <= 5 and all(g.startswith("the man stro") for g in got)
    assert engine.unknown_words("a persn jumsp") == ["persn", "jumsp"]
    assert "person" in engine.nearest("persn") and "jump" in engine.nearest("jumsp")


def test_typing_cost():
    """Reported: the work per key (the ghost; uncached), the word list at the pause, and making the help at load."""
    start()
    sentence = "a person walks forward and sits down on a chair"
    t0 = time.perf_counter()
    engine.unload()
    engine.model()
    load = time.perf_counter() - t0
    helper = engine.helper()
    lane.begin(lane._s.typing[0], "")
    keys, pause = [], []
    for ch in sentence:
        helper.cache.clear()
        t0 = time.perf_counter()
        write(ch)
        keys.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        paused(lane._s.help)
        pause.append(time.perf_counter() - t0)
    fresh = []
    for k in range(len(sentence)):
        helper.cache.clear()
        t0 = time.perf_counter()
        helper.sentence(sentence[: k + 1])
        fresh.append(time.perf_counter() - t0)
    ms = [1000 * max(keys), 1000 * sum(keys) / len(keys), 1000 * max(fresh), 1000 * max(pause)]
    print(f"> typing: per key {ms[1]:.2f} ms mean, {ms[0]:.2f} ms max (a new ghost {ms[2]:.2f} ms max); "
          f"list {ms[3]:.2f} ms; model + help load {1000 * load:.0f} ms")  # fmt: skip
    assert os.environ.get("CI") or ms[0] < 16, f"a key took {ms[0]:.1f} ms, more than a frame"
