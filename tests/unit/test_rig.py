import pytest
import rigs
from sprongmotion import rig


class V(tuple):
    x = property(lambda self: self[0])


class Bone:
    def __init__(self, name, head, tail, parent):
        self.name, self.head_local, self.tail_local, self.use_deform = name, V(head), V(tail), True
        self.parent, self.children = parent, []
        if parent:
            parent.children.append(self)

    @property
    def parent_recursive(self):
        return [self.parent, *self.parent.parent_recursive] if self.parent else []


class Bones(list):
    def __getitem__(self, key):
        return self.get(key) if isinstance(key, str) else super().__getitem__(key)

    def get(self, key):
        return next((b for b in self if b.name == key), None)


def armature(spec):
    out = Bones()
    for name, (parent, head, tail) in spec.items():
        out.append(Bone(name, head, tail, out.get(parent)))
    return type("Armature", (), {"bones": out})()


@pytest.mark.parametrize("name", rigs.RIGS)
def test_mapping(name):
    want = rigs.expected(rigs.RIGS[name])
    assert rig.auto_match(armature(rigs.bones(rigs.RIGS[name]))) == want
    assert rig.match_structure(armature(rigs.bones(rigs.RIGS[name], junk=False))) == want


def test_rest_joints_stand_on_the_floor():
    import numpy as np

    assert rig.rest_joints(np.tile([0, 0.1, 0], (22, 1)))[:, 1].min() == 0
