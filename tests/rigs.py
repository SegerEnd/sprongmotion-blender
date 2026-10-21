"""Test rigs as data, for both test layers: per rig its bones (name: parent, rest head, rest tail, Z up, metres unless
data says otherwise), the armature-data and object transforms, and the bone map the add-on must find. Bones named
.L / Left get their .R / Right mirror. JUNK bones (accessories, cages, face bones) must never be keyed."""

import math

JUNK = {
    "Handle": ("Head", (0.1, 0.1, 0.1), (0.1, 0.3, 0.1)),
    "DynamicHead": ("Head", (0, 0.1, 0), (0, 0.5, 0)),
    "Jaw": ("DynamicHead", (0, 0.1, 0.2), (0, 0.05, 0.4)),
    "Torso_OuterCage": (None, (0, -3, 0), (0, 3, 0)),
}
_TURN = (math.cos(0.25), 0, 0, math.sin(0.25))  # an object yaw of 0.5 rad, quaternion w x y z
_Y_UP_CM = ((100, 0, 0), (0, 0, 100), (0, -100, 0))  # what FBX imports make of a Z-up rig in metres

RIGS = {
    "simple16": {  # one spine bone, no collars, no toes, an A pose, short legs and long arms
        "bones": {
            "hips": (None, (0, 0, 0.5), (0, 0, 0.58)),
            "body": ("hips", (0, 0, 0.58), (0, 0, 0.95)),
            "neck_bone": ("body", (0, 0, 0.95), (0, 0, 1.0)),
            "head_bone": ("neck_bone", (0, 0, 1.0), (0, 0, 1.3)),
            "arm_up.L": ("body", (0.12, 0, 0.9), (0.4, 0, 0.7)),
            "arm_low.L": ("arm_up.L", (0.4, 0, 0.7), (0.62, -0.02, 0.5)),
            "hand_x.L": ("arm_low.L", (0.62, -0.02, 0.5), (0.72, -0.02, 0.42)),
            "leg_up.L": ("hips", (0.09, 0, 0.5), (0.1, 0, 0.27)),
            "leg_low.L": ("leg_up.L", (0.1, 0, 0.27), (0.1, 0.01, 0.05)),
            "foot_x.L": ("leg_low.L", (0.1, 0.01, 0.05), (0.1, -0.12, 0.03)),
        },
        "map": {
            "pelvis": "hips",
            "spine1": "body",
            "neck": "neck_bone",
            "head": "head_bone",
            "l_shoulder": "arm_up.L",
            "l_elbow": "arm_low.L",
            "l_wrist": "hand_x.L",
            "l_hip": "leg_up.L",
            "l_knee": "leg_low.L",
            "l_ankle": "foot_x.L",
        },
    },
    "mixamo": {  # T pose, imported from FBX: Y up in centimetres, the object turned upright at 0.01 scale
        "bones": {
            "Hips": (None, (0, 0, 1.0), (0, 0, 1.1)),
            "Spine": ("Hips", (0, 0, 1.1), (0, 0, 1.22)),
            "Spine1": ("Spine", (0, 0, 1.22), (0, 0, 1.36)),
            "Spine2": ("Spine1", (0, 0, 1.36), (0, 0, 1.5)),
            "Neck": ("Spine2", (0, 0, 1.5), (0, 0, 1.6)),
            "Head": ("Neck", (0, 0, 1.6), (0, 0, 1.8)),
            "LeftShoulder": ("Spine2", (0.03, 0, 1.45), (0.17, 0, 1.45)),
            "LeftArm": ("LeftShoulder", (0.17, 0, 1.45), (0.45, 0, 1.45)),
            "LeftForeArm": ("LeftArm", (0.45, 0, 1.45), (0.71, 0, 1.45)),
            "LeftHand": ("LeftForeArm", (0.71, 0, 1.45), (0.8, 0, 1.45)),
            "LeftUpLeg": ("Hips", (0.09, 0, 0.95), (0.09, 0, 0.52)),
            "LeftLeg": ("LeftUpLeg", (0.09, 0, 0.52), (0.09, 0, 0.1)),
            "LeftFoot": ("LeftLeg", (0.09, 0, 0.1), (0.09, -0.12, 0.02)),
            "LeftToeBase": ("LeftFoot", (0.09, -0.12, 0.02), (0.09, -0.2, 0.02)),
        },
        "prefix": "mixamorig:",
        "data": _Y_UP_CM,
        "rotation": (math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0),
        "scale": 0.01,
        "map": {
            "pelvis": "Hips",
            "spine1": "Spine",
            "spine2": "Spine1",
            "spine3": "Spine2",
            "neck": "Neck",
            "head": "Head",
            "l_collar": "LeftShoulder",
            "l_shoulder": "LeftArm",
            "l_elbow": "LeftForeArm",
            "l_wrist": "LeftHand",
            "l_hip": "LeftUpLeg",
            "l_knee": "LeftLeg",
            "l_ankle": "LeftFoot",
            "l_foot": "LeftToeBase",
        },
    },
    "r15": {  # the official R15 template's bones (Y up, studs), its object turned upright, plus avatar junk
        "bones": {
            "Root": (None, (0, 0, 0), (0, 0.01, 0)),
            "HumanoidRootNode": ("Root", (0, 0, 0), (0, 0.01, 0)),
            "LowerTorso": ("HumanoidRootNode", (0, 0, 0), (0, 0.331, 0)),
            "UpperTorso": ("LowerTorso", (0, 0.305, 0.056), (0, 1.54, 0.056)),
            "Head": ("UpperTorso", (0, 1.663, 0.02), (0, 2.897, 0.02)),
            "LeftUpperArm": ("UpperTorso", (0.47, 1.379, 0.012), (0.885, 0.809, 0.012)),
            "LeftLowerArm": ("LeftUpperArm", (0.885, 0.809, 0.012), (1.35, 0.121, 0.193)),
            "LeftHand": ("LeftLowerArm", (1.35, 0.121, 0.193), (1.583, -0.223, 0.283)),
            "LeftUpperLeg": ("LowerTorso", (0.312, -0.123, 0.063), (0.494, -1.436, 0.021)),
            "LeftLowerLeg": ("LeftUpperLeg", (0.494, -1.436, 0.021), (0.651, -2.752, -0.074)),
            "LeftFoot": ("LeftLowerLeg", (0.651, -2.922, -0.074), (0.651, -2.922, 0.426)),
        },
        "junk": JUNK,
        "rotation": (math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0),
        "map": {
            "pelvis": "LowerTorso",
            "spine1": "UpperTorso",
            "neck": "Head",
            "l_shoulder": "LeftUpperArm",
            "l_elbow": "LeftLowerArm",
            "l_wrist": "LeftHand",
            "l_hip": "LeftUpperLeg",
            "l_knee": "LeftLowerLeg",
            "l_ankle": "LeftFoot",
        },
    },
    "r6": {  # one bone per limb, all from the torso; studs at 0.28 m (the import scale is a guess)
        "bones": {
            "HumanoidRootPart": (None, (0, 0, 3), (0, 0, 3.5)),
            "Torso": ("HumanoidRootPart", (0, 0, 3), (0, 0, 4.6)),
            "Head": ("Torso", (0, 0, 4.6), (0, 0, 5.6)),
            "Left Arm": ("Torso", (1.5, 0, 4.4), (1.5, 0, 2.5)),
            "Left Leg": ("Torso", (0.5, 0, 3), (0.5, 0, 1)),
        },
        "junk": JUNK,
        "rotation": _TURN,
        "scale": 0.28,
        "map": {"pelvis": "Torso", "neck": "Head", "l_shoulder": "Left Arm", "l_hip": "Left Leg"},
    },
}


def right(name):
    return name.replace(".L", ".R").replace("Left", "Right")


def bones(rig, junk=True):
    """name -> (parent, head, tail) with mirrors, prefix and junk, in armature space (data transform applied)."""
    c, pre = rig.get("centre", 0), rig.get("prefix", "")
    data = rig.get("data", ((1, 0, 0), (0, 1, 0), (0, 0, 1)))

    def place(p, mirror=False):
        p = (2 * c - p[0], p[1], p[2]) if mirror else p
        return tuple(sum(r[i] * p[i] for i in range(3)) for r in data)

    out = {}
    for name, (parent, head, tail) in rig["bones"].items():
        for mirror in {False, right(name) != name}:
            n, par = (right(name), parent and right(parent)) if mirror else (name, parent)
            out[pre + n] = (par and pre + par, place(head, mirror), place(tail, mirror))
    return out | (rig.get("junk", {}) if junk else {})


def expected(rig):
    """The joint -> bone map the add-on must find."""
    pre = rig.get("prefix", "")
    return {j: pre + n for j, n in rig["map"].items()} | {
        "r" + j[1:]: pre + right(n) for j, n in rig["map"].items() if j.startswith("l_")
    }
