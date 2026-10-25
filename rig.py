"""The 22-joint skeleton, matching a rig's bones to it (presets, else by structure) and the add-on's own rig."""

import bpy
import numpy as np

JOINTS = [
    "pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle", "r_ankle", "spine3", "l_foot",
    "r_foot", "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist",
    "r_wrist",
]  # fmt: skip
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
CHILD = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, -1, 15, 16, 17, -1, 18, 19, 20, 21, -1, -1]
ARM = ["collar", "shoulder", "elbow", "wrist"]
LEG = ["hip", "knee", "ankle", "foot"]
LEGS = ((1, 4, 7), (2, 5, 8))  # hip, knee, ankle: left, right
ARMS = ((16, 18, 20), (17, 19, 21))  # shoulder, elbow, wrist: left, right


def _left_right(side, name):
    return ("Left" if side == "L" else "Right") + name


def _sided(names, fmt):
    return {f"{s}_{j}": fmt(side, n) for j, n in names.items() for s, side in (("l", "L"), ("r", "R"))}


_MIXAMO = dict(zip(ARM + LEG, ["Shoulder", "Arm", "ForeArm", "Hand", "UpLeg", "Leg", "Foot", "ToeBase"], strict=True))
_RIGIFY = dict(
    zip(ARM + LEG, ["shoulder", "upper_arm_fk", "forearm_fk", "hand_fk", "thigh_fk", "shin_fk", "foot_fk", "toe_fk"],
        strict=True)
)  # fmt: skip
_R15 = dict(zip(ARM[1:] + LEG[:3], ["UpperArm", "LowerArm", "Hand", "UpperLeg", "LowerLeg", "Foot"], strict=True))
PRESETS = {
    "OWN": {j: j for j in JOINTS},
    "MIXAMO": {
        "pelvis": "Hips",
        "spine1": "Spine",
        "spine2": "Spine1",
        "spine3": "Spine2",
        "neck": "Neck",
        "head": "Head",
        **_sided(_MIXAMO, _left_right),
    },  # fmt: skip
    "R15": {"pelvis": "LowerTorso", "spine1": "UpperTorso", "neck": "Head", **_sided(_R15, _left_right)},
    "R6": {"pelvis": "Torso", "neck": "Head", **_sided({"shoulder": " Arm", "hip": " Leg"}, _left_right)},
    "RIGIFY": {
        "pelvis": "torso",
        "spine2": "spine_fk.002",
        "spine3": "spine_fk.003",
        "neck": "neck",
        "head": "head",
        **_sided(_RIGIFY, lambda side, n: f"{n}.{side}"),
    },  # fmt: skip
}
PRESET_ITEMS = [
    ("AUTO", "Auto", "Known rigs by name, any other by the shape of its hierarchy"),
    ("OWN", "SprongMotion Rig", "The add-on's own 22-bone rig"),
    ("MIXAMO", "Mixamo", "mixamorig:Hips, LeftUpLeg, ..."),
    ("R15", "Roblox R15", "LowerTorso, UpperTorso, LeftUpperArm, ..."),
    ("R6", "Roblox R6", "Torso, Head, Left Arm, ..."),
    ("RIGIFY", "Rigify", "The FK controls of a generated Rigify rig (arms and legs switched to FK)"),
]


def match_preset(arm, preset):
    """Joint -> bone name for a preset, accepting any 'prefix:' before the names (mixamorig1:, ...)."""
    names = {b.name.split(":")[-1]: b.name for b in arm.bones}
    return {j: names[b] for j, b in PRESETS[preset].items() if b in names}


def auto_match(arm):
    """A preset whose names all exist (Roblox parts are exact), else the best one with 12+ joints, else structure."""
    found = {p: match_preset(arm, p) for p in PRESETS}
    complete = [p for p in PRESETS if len(found[p]) == len(PRESETS[p])]
    best = max(complete or PRESETS, key=lambda p: len(found[p]))
    return found[best] if complete or len(found[best]) >= 12 else match_structure(arm)


def match_structure(arm):
    """Match by hierarchy: the pelvis has two chains going down and a spine going up. +x is the character's left."""
    bones = [b for b in arm.bones if b.use_deform] or list(arm.bones)
    if not bones:
        return {}
    for up in (np.array([0, 0, 1.0]), np.array([0, 1.0, 0])):
        found = _Shape(bones, up).match()
        if found:
            return found
    return {}


class _Shape:
    def __init__(self, bones, up):
        self.bones, self.keep, self.up = bones, {b.name for b in bones}, up
        pts = np.array([p for b in bones for p in (b.head_local, b.tail_local)])
        self.size = np.ptp(pts @ up)

    def kids(self, b):
        return [c for c in b.children if c.name in self.keep]

    def subtree(self, b):
        return [b, *[d for c in self.kids(b) for d in self.subtree(c)]]

    def reach(self, b):
        h = [np.array(p) @ self.up for d in self.subtree(b) for p in (d.head_local, d.tail_local)]
        return min(h), max(h)

    def side(self, b, centre):
        x = np.mean([d.head_local.x for d in self.subtree(b)]) - centre
        return "l" if x > 0.05 * self.size else "r" if x < -0.05 * self.size else ""

    def chain(self, b):
        out = [b]
        while len(self.kids(out[-1])) == 1:
            out.append(self.kids(out[-1])[0])
        return out

    def match(self):
        for p in sorted(self.bones, key=lambda b: len(b.parent_recursive)):
            found = self.from_pelvis(p)
            if found:
                return found
        return {}

    def from_pelvis(self, p):
        h, x = np.array(p.head_local) @ self.up, p.head_local.x
        down = [c for c in self.kids(p) if self.reach(c)[0] < h - 0.25 * self.size and self.side(c, x)]
        ups = [c for c in self.kids(p) if self.reach(c)[1] > h + 0.25 * self.size]
        if len({self.side(c, x) for c in down}) < 2 or not ups:
            return None
        found = {"pelvis": p.name}
        for c in sorted(down, key=lambda c: self.reach(c)[0], reverse=True):
            found.update(_limb_joints(self.side(c, x), LEG, self.chain(c)))
        top = max(ups, key=lambda c: self.reach(c)[1])
        arms = [c for c in self.kids(p) if self.side(c, x) and c not in down]
        if len({self.side(c, x) for c in arms}) == 2:  # one torso bone for pelvis and chest (Roblox R6)
            found.update(self.arms(arms, x), neck=top.name)
            return found
        found.update(self.spine(top, x))
        return found

    def spine(self, b, x):
        spine = [b]
        while True:
            kids = self.kids(spine[-1])
            sides = [c for c in kids if self.side(c, x)]
            if len({self.side(c, x) for c in sides}) == 2 or not kids:
                break
            spine.append(max(kids, key=lambda c: self.reach(c)[1]))
        names = {1: ["spine1"], 2: ["spine1", "spine3"]}.get(len(spine), ["spine1", "spine2", "spine3"])
        picks = [spine[0], spine[-1]] if len(spine) == 2 else [spine[0], spine[len(spine) // 2], spine[-1]]
        found = dict(zip(names, (s.name for s in picks), strict=False))
        top = spine[-1]
        found.update(self.arms([c for c in self.kids(top) if self.side(c, x)], x))
        centre = [c for c in self.kids(top) if not self.side(c, x)]
        if centre:
            neck = max(centre, key=lambda c: self.reach(c)[1])
            found["neck"] = neck.name
            if self.kids(neck):
                found["head"] = max(self.kids(neck), key=lambda c: self.reach(c)[1]).name
        return found

    def arms(self, kids, x):
        found = {}
        for c in sorted(kids, key=lambda c: len(self.subtree(c))):
            found.update(_limb_joints(self.side(c, x), ARM, self.chain(c)))
        return found


def _limb_joints(side, names, chain):
    """A chain onto a limb's joints from the shoulder or hip: only a 4-bone arm has a collar, a 4-bone leg a toe."""
    names = names if names is LEG or len(chain) >= len(names) else names[1:]
    return {f"{side}_{n}": b.name for n, b in zip(names, chain[: len(names)], strict=False)}


def fill_map(obj):
    s = obj.sprongmotion
    found = auto_match(obj.data) if s.preset == "AUTO" else match_preset(obj.data, s.preset)
    while len(s.bones) > len(JOINTS):
        s.bones.remove(len(s.bones) - 1)
    while len(s.bones) < len(JOINTS):  # filled in place, not cleared: a half-written map never stays behind
        s.bones.add()
    for item, j in zip(s.bones, JOINTS, strict=True):
        item.joint, item.bone = j, found.get(j, "")


def ensure_map(obj):
    """Fill the bone map when it is missing or half written (an interrupted fill)."""
    if len(obj.sprongmotion.bones) != len(JOINTS):
        fill_map(obj)


def bone_map(obj):
    bones = obj.data.bones
    return {JOINTS.index(b.joint): b.bone for b in obj.sprongmotion.bones if b.bone in bones and b.joint in JOINTS}


RIGIFY_SWITCHES = ("upper_arm_parent.L", "upper_arm_parent.R", "thigh_parent.L", "thigh_parent.R", "torso")


def prepare(obj):
    """Rigify: limbs on FK, and no half-way follow blends, which the pose maths cannot express."""
    switches = {"IK_FK": 1.0, "neck_follow": 1.0, "head_follow": 1.0}
    for name in RIGIFY_SWITCHES:
        pb = obj.pose.bones.get(name)
        for key in switches if pb is not None else ():
            if key in pb:
                pb[key] = switches[key]


def to_blender(v):
    """The model's frame (Y up, facing +Z) to Blender's (Z up, facing -Y)."""
    v = np.asarray(v)
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], -1)


def rest_joints(offsets):
    """Joint positions of the reference skeleton at rest, model frame, the lowest joint at y = 0."""
    p = np.zeros((len(JOINTS), 3))
    for j, par in enumerate(PARENTS):
        if par >= 0:
            p[j] = p[par] + offsets[j]
    p[:, 1] -= p[:, 1].min()
    return p


def own_rig_joints(offsets):
    """Rest positions for the add-on's own rig in Blender's frame: the reference bone lengths, arms in an A pose."""
    p = to_blender(rest_joints(offsets))
    a = np.radians(40)
    for side in (1, -1):
        chain = [JOINTS.index(("l_" if side > 0 else "r_") + n) for n in ARM[1:]]
        d = np.array([side * np.cos(a), 0, -np.sin(a)])
        lengths = [np.linalg.norm(offsets[j]) for j in chain]
        p[chain[0]] = p[PARENTS[chain[0]]] + np.array([side, 0, 0]) * lengths[0]
        for k in (1, 2):
            p[chain[k]] = p[chain[k - 1]] + d * lengths[k]
    return p


def build_own_rig(context, offsets, name="SprongMotion Rig"):
    p = own_rig_joints(offsets)
    arm = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, arm)
    obj.location = context.scene.cursor.location
    context.collection.objects.link(obj)
    context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for j, joint in enumerate(JOINTS):
        e = arm.edit_bones.new(joint)
        e.head = p[j]
        e.tail = p[CHILD[j]] if CHILD[j] >= 0 else p[j] + _leaf_direction(p, j)
    for j, par in enumerate(PARENTS):
        if par >= 0:
            e = arm.edit_bones[JOINTS[j]]
            e.parent, e.use_connect = arm.edit_bones[JOINTS[par]], CHILD[par] == j
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.show_in_front = True
    return obj


def _leaf_direction(p, j):
    if JOINTS[j] == "head":
        return np.array([0, 0, 0.18])
    d = p[j] - p[PARENTS[j]]
    return d / np.linalg.norm(d) * (0.08 if "wrist" in JOINTS[j] else 0.05)
