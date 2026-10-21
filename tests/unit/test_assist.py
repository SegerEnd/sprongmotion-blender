from types import SimpleNamespace

import numpy as np
from sprongmotion import assist

SENTENCES = ["a person sits down on a chair", "a person sits down on the floor", "a person walks forward slowly"]
WORDS = ["chair", "chairs", "floor", "forward", "slowly", "stroll", "strolls", "strong", "walks"]


def encode(text):
    """A bag of words: the cosine counts shared words."""
    return np.array([float(w in text.split()) for w in ("chair", "floor", "walks", "slowly", "sofa", "ground")])


def helper():
    return assist.Help(SENTENCES, WORDS, encode)


def test_ghost_is_the_best_sentence_rest():
    h = helper()
    assert h.ghost("a person sits down on the ground") == ""
    assert h.ghost("a person sits down on the f") == "loor"
    assert h.ghost("A person sits d") in {"own on a chair", "own on the floor"}
    assert h.ghost("a person walks") == " forward slowly"
    assert h.ghost("") == h.ghost("a pe") == "", "nothing for empty or short text"
    assert h.ghost("a person walks forward slowly") == "", "a whole sentence has no rest"


def test_ranking_follows_the_encoder():
    h = assist.Help(["a man sits on a chair", "a man sits on the floor"], WORDS, encode)
    h.encode = lambda text: encode(text.replace("a man sits", "a man sits floor"))  # leans the typed text to "floor"
    assert h.sentence("a man sits") == "a man sits on the floor"


def test_the_shown_sentence_stays_while_it_fits():
    h = helper()
    shown = h.sentence("a person sits d")
    other = next(s for s in SENTENCES[:2] if s != shown)
    assert h.sentence("a person sits do", shown) == shown
    assert h.sentence(other[:-1], shown) == other, "typing past it picks again"


def test_word_completion():
    h = helper()
    assert h.complete("a person str", 12) == ["stroll", "strong", "strolls"], "shorter first"
    assert h.complete("a person s", 10) == [], "two letters at least"
    assert h.complete("a person walks", 14) == [], "a known word is left alone"
    assert h.complete("a person strxx", 14) == []
    assert h.complete("a strxl", 5) == [], "not inside a word"
    assert h.complete("the ch", 6) == ["chair", "chairs"], "our sentences' words first"
    assert len(assist.Help(SENTENCES, [f"wa{i:03}" for i in range(50)], encode).complete("wa", 2)) == assist.ROWS


def test_vocab_filter_keeps_only_known_words():
    known = set(assist.lines("words.txt"))
    assert {"walks", "chair", "pirouette"} - known == {"pirouette"}
    assert not {"aceous", "zzle", "tion", "ing"} & known, "word endings are no words"
    assert all(set(s.split()) <= known | {w for w in s.split() if len(w) < 4} for s in assist.lines("sentences.txt"))


def test_near_words():
    assert helper().near("chiar") == ["chair"]


def key(type_, ctrl=False):
    return SimpleNamespace(type=type_, ctrl=ctrl, oskey=False)


def test_keys():
    k = assist.kind
    assert [k(key(t)) for t in ("TAB", "RET", "ESC", "UP_ARROW", "DOWN_ARROW", "A")] == [
        "tab", "enter", "esc", "up", "down", None
    ]  # fmt: skip
    assert k(key("RIGHT_ARROW", ctrl=True)) == "word" and k(key("RIGHT_ARROW")) is None
    t = "a person sits d"
    assert assist.take("tab", t, 15, "own on a chair", [], 0) == ("a person sits down on a chair", 29, 0, False)
    assert assist.take("word", t, 15, "own on a chair", [], 0) == ("a person sits down", 18, 0, False)
    assert assist.take("word", "a person sits down", 18, " on a chair", [], 0)[0] == "a person sits down on"
    assert assist.take("esc", t, 15, "own", [], 0) == (t, 15, 0, True)
    assert assist.take("enter", t, 15, "own", [], 0) is None, "Enter ends the edit, it does not take the ghost"
    assert assist.take("esc", t, 15, "", [], 0) is None and assist.take("tab", t, 15, "", [], 0) is None
    rows = ["stroll", "strolls"]
    assert assist.take("down", "a str", 5, "", rows, 0) == ("a str", 5, 1, False)
    assert assist.take("up", "a str", 5, "", rows, 0)[2] == 1
    assert assist.take("enter", "a str now", 5, "", rows, 1) == ("a strolls now", 9, 0, False)
    assert assist.take("tab", "a str", 5, "", rows, 0)[:2] == ("a stroll", 8)
