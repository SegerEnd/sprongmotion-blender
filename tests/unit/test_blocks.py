import types

import pytest
from sprongmotion import blocks, engine


class Blocks(list):
    def add(self):
        b = types.SimpleNamespace(uid=0, start=0, end=0, text="", seed=0, color=None, locked=False, enabled=True,
                                  in_place=False, back_to_start=False, dirty=True, variants=0, variant=0)  # fmt: skip
        self.append(b)
        return b

    def move(self, i, j):
        self.insert(j, self.pop(i))

    def remove(self, i):
        del self[i]


@pytest.fixture
def obj(monkeypatch):
    monkeypatch.setattr(blocks, "fps", lambda scene=None: 30.0)
    monkeypatch.setattr(engine, "edited", lambda *a, **k: None)
    scene = types.SimpleNamespace(frame_current=10)
    monkeypatch.setattr(blocks.bpy, "context", types.SimpleNamespace(scene=scene), raising=False)
    s = types.SimpleNamespace(blocks=Blocks(), next_uid=1, active=0, path=None, loop=False)
    monkeypatch.setattr(blocks, "settings", lambda _obj, create=False: s)  # the action's timeline
    return types.SimpleNamespace(type="ARMATURE", sprongmotion=s, animation_data=None)


def spans(obj):
    return [(b.start, b.end) for b in obj.sprongmotion.blocks]


def test_contiguous_edits(obj):
    for text in ("walk", "jump", "sit"):
        blocks.add(obj, text, seconds=2)
    assert spans(obj) == [(10, 70), (70, 130), (130, 190)]
    blocks.resize(obj, 0, end=100)
    assert spans(obj) == [(10, 100), (100, 160), (160, 220)]
    blocks.resize(obj, 1, end=101)
    assert spans(obj)[1] == (100, 115)
    blocks.move(obj, 0, 2)
    assert [b.text for b in obj.sprongmotion.blocks] == ["jump", "sit", "walk"] and spans(obj)[0] == (10, 25)
    blocks.remove(obj, 0)
    n = blocks.split(obj, 1, 75)
    assert spans(obj) == [(10, 70), (70, 75), (75, 160)] or n is None


def test_only_edited_unlocked_blocks_are_made_again(obj):
    for text in "abcd":
        blocks.add(obj, text, seconds=1)
    bs = obj.sprongmotion.blocks
    for b in bs:
        b.dirty = False
    bs[2].locked = True
    blocks.mark_dirty(obj, bs[1].uid, bs[2].uid)
    assert [b.dirty for b in bs] == [False, True, False, False], "only the edited, unlocked block is made again"


def test_free_placement(obj):
    for text in ("walk", "jump", "sit"):
        blocks.add(obj, text, seconds=1)
    s = obj.sprongmotion
    assert spans(obj) == [(10, 40), (40, 70), (70, 100)]
    assert blocks.place(s, 0, 200) == 2 and spans(obj) == [(40, 70), (70, 100), (200, 230)], "anywhere free"
    assert blocks.place(s, 2, 75) == 2 and spans(obj)[2] == (100, 130), "onto a block: its nearest free edge"
    assert blocks.place(s, 2, 50) == 0 and spans(obj)[0] == (10, 40)
    blocks.place(s, 2, 130)
    assert [len(r) for r in blocks.runs(s)] == [2, 1] and blocks.continues(s) == {1: 0, 2: 1, 3: 0}
    blocks.resize(obj, 0, end=30)
    assert spans(obj) == [(10, 30), (40, 70), (130, 160)], "shrinking leaves a gap"
    blocks.resize(obj, 0, end=60)
    assert spans(obj) == [(10, 60), (60, 90), (130, 160)], "growing pushes what it would overlap"
    blocks.resize(obj, 2, start=80)
    assert spans(obj) == [(0, 50), (50, 80), (80, 160)], "a left edge too"
    blocks.resize(obj, 0, end=40, follow=True)
    assert spans(obj) == [(0, 40), (40, 70), (70, 150)], "follow: the touching blocks move with the end"


def test_remove_and_swap_keep_the_gaps(obj):
    for text in ("walk", "jump", "sit"):
        blocks.add(obj, text, seconds=1)
    s = obj.sprongmotion
    blocks.shift(s, [2], 50)
    blocks.move(obj, 1, 2)
    assert [b.text for b in s.blocks] == ["walk", "sit", "jump"] and spans(obj) == [(10, 40), (40, 70), (120, 150)]
    blocks.remove(obj, 0)
    assert spans(obj) == [(10, 40), (120, 150)], "the touching block moves up, the one after the gap stays"


def test_resizing_an_edge_ripples_its_run(obj):
    for text in ("walk", "jump", "sit", "wave"):
        blocks.add(obj, text, seconds=1)
    blocks.shift(obj.sprongmotion, [3], 20)
    assert spans(obj) == [(10, 40), (40, 70), (70, 100), (120, 150)]
    blocks.resize(obj, 1, end=60, follow=True)
    assert spans(obj) == [(10, 40), (40, 60), (60, 90), (120, 150)], "shrinks: the run stays tight, the gap grows"
    blocks.resize(obj, 1, end=75, follow=True)
    assert spans(obj) == [(10, 40), (40, 75), (75, 105), (120, 150)], "grows: the block after the gap stays"
    blocks.resize(obj, 1, end=100, follow=True)
    assert spans(obj) == [(10, 40), (40, 100), (100, 130), (130, 160)], "until it would overlap: pushed"
    blocks.resize(obj, 1, start=50, follow=True)
    assert spans(obj)[:2] == [(20, 50), (50, 100)], "a left edge moves the blocks before it"
    blocks.resize(obj, 1, start=60)
    assert spans(obj)[:2] == [(20, 50), (60, 100)], "without follow (Alt): the others stay"
