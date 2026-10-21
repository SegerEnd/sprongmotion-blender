"""Keys on the action: a take is written as a few Bezier keys that look the same as a key on every frame (unbaked),
or as every frame. Key values are numpy arrays [frames, channels]; a group is one bone's quaternion or location."""

import numpy as np

KEYFRAME, GENERATED = 0, 5  # KeyframeType values
LINEAR, BEZIER, ALIGNED = 1, 2, 3  # interpolation and handle-type values (foreach_set takes the C values)


def fit(x, y, m, keys):
    """The Bezier through each curve's keys: y and m [F, C] values and slopes at frames x, keys [F, C] bool (first and
    last frame keyed); handles a third of the way to each neighbour, as write sets them."""
    n, idx, c = len(x), np.arange(len(x))[:, None], np.arange(y.shape[1])
    p = np.maximum.accumulate(np.where(keys[:-1], idx[:-1], 0), 0)
    p = np.vstack([p, p[-1:]])  # the last frame ends the last span
    q = np.minimum.accumulate(np.where(keys, idx, n - 1)[::-1], 0)[::-1][p + 1, c]
    h = x[q] - x[p]
    t = (x[:, None] - x[p]) / h
    t2, t3 = t * t, t * t * t
    yp, yq, mp, mq = y[p, c], y[q, c], m[p, c], m[q, c]
    return (2 * t3 - 3 * t2 + 1) * yp + (3 * t2 - 2 * t3) * yq + h * ((t3 - 2 * t2 + t) * mp + (t3 - t2) * mq)


def angle(fit, y):
    """Rotation angle between quaternion rows (w x y z)."""
    fit = fit / np.linalg.norm(fit, axis=1, keepdims=True)
    return 2 * np.arccos(np.clip(np.abs((fit * y).sum(1)), 0, 1))


def distance(fit, y):
    return np.linalg.norm(fit - y, axis=1)


def extremes(y, size, width=5):
    """[F, C] mask: frames where a channel turns by more than size (one or [C]), after a light box smoothing (jitter is
    not a turn). A turn compares with the turns (or ends) before and after it in its own channel."""
    box = np.ones(width) / width
    s = np.column_stack([np.convolve(np.pad(c, width // 2, mode="edge"), box, mode="valid") for c in y.T])
    turn = np.zeros(s.shape, bool)
    turn[1:-1] = np.diff(np.sign(np.diff(s, axis=0)), axis=0) != 0
    ends = np.zeros(s.shape, bool)
    ends[[0, -1]] = True
    c, f = np.nonzero((turn | ends).T)  # by channel, then frame: neighbours in a channel are neighbours here
    v, size = s[f, c], np.broadcast_to(size, s.shape[1])[c]
    inner = np.flatnonzero(turn[f, c])
    big = inner[(np.abs(v[inner] - v[inner - 1]) > size[inner]) | (np.abs(v[inner] - v[inner + 1]) > size[inner])]
    out = np.zeros(s.shape, bool)
    out[f[big], c[big]] = True
    return out


def _refine(keys, error):
    """Add the worst frame of every span between keys still off by more than 1 (error(keys, curves) [F, curves],
    normalised by its limit) to keys [F, C], until no span is. A curve without new keys is done."""
    keys, cols = keys.copy(), np.arange(keys.shape[1])
    while len(cols):
        e = error(keys[:, cols], cols)
        f, c = np.nonzero(e > 1)
        seg = np.cumsum(keys[:, cols], 0)[f, c]
        order = np.lexsort((-e[f, c], seg, c))
        f, c, seg = f[order], c[order], seg[order]
        first = np.ones(len(f), bool)
        first[1:] = (c[1:] != c[:-1]) | (seg[1:] != seg[:-1])
        keys[f[first], cols[c[first]]] = True
        cols = cols[np.unique(c)]
    return keys


def unbake(x, groups, marks, shared=False):
    """groups: [(values [F, C], error function, limit)] -> (key indices per group and curve, slopes per group).
    Each curve gets its own keys within limit / 2, then frames where the bone's angle (or the root's distance) is
    still over the limit get a key on all its curves; shared: one key set for all bones. Handles take the take's own
    slope at each key. All curves at once: keys are a mask [frames, curves]."""
    y = np.concatenate([g[0] for g in groups], 1)
    m, n, cols = np.gradient(y, x, axis=0), len(x), np.cumsum([0] + [g[0].shape[1] for g in groups])
    spans = [slice(a, b) for a, b in zip(cols[:-1], cols[1:], strict=True)]
    parts = [(g[1], s, np.broadcast_to(np.reshape(g[2], (-1, 1)), (n, 1))) for g, s in zip(groups, spans, strict=True)]
    seeds = np.zeros((n, 1), bool)
    seeds[[0, -1]], seeds[np.searchsorted(x, marks)] = True, True
    if shared:

        def worst(k):
            e = fit(x, y, m, np.repeat(k, y.shape[1], 1))
            return np.max([fn(e[:, s], y[:, s]) / lim[:, 0] for fn, s, lim in parts], 0)

        k = np.flatnonzero(_refine(seeds, lambda k, _: worst(k)[:, None])[:, 0])
        return [[k] * g[0].shape[1] for g in groups], [m[:, s] for s in spans]
    half = np.concatenate([np.repeat(lim, s.stop - s.start, 1) / 2 for _, s, lim in parts], 1)
    keys = _refine(
        seeds | extremes(y, half.max(0)), lambda k, c: np.abs(fit(x, y[:, c], m[:, c], k) - y[:, c]) / half[:, c]
    )
    while True:  # a keyed frame cannot get closer: only frames between keys are looked at
        e, add = fit(x, y, m, keys), np.zeros_like(keys)
        for fn, s, lim in parts:
            err = np.where(keys[:, s].all(1), 0, fn(e[:, s], y[:, s]) / lim[:, 0])
            bad = np.flatnonzero(err > 1.0001)
            runs = np.split(bad, np.flatnonzero(np.diff(bad) > 1) + 1) if len(bad) else []
            add[[r[np.argmax(err[r])] for r in runs], s] = True
        if not add.any():
            break
        keys = keys | add
    return [[np.flatnonzero(keys[:, c]) for c in range(s.start, s.stop)] for s in spans], [m[:, s] for s in spans]


def get(kp):
    co, kind = np.zeros(2 * len(kp), np.float32), np.zeros(len(kp), np.int32)
    kp.foreach_get("co", co)
    kp.foreach_get("type", kind)
    return co.reshape(-1, 2), kind


def remove(kp, drop):
    if drop.all():
        kp.clear()
    else:
        for i in np.flatnonzero(drop)[::-1]:
            kp.remove(kp[int(i)], fast=True)


def write(fc, x, y, m=None, kind=GENERATED, spans=None):
    """Keys at frames x with values y (slopes m: Bezier with aligned handles, else linear). Existing keys inside spans
    [(lo, hi)] and GENERATED keys anywhere go; the user's own keys outside them stay."""
    kp = fc.keyframe_points
    co, kinds = get(kp)
    inside = kinds == GENERATED
    for lo, hi in spans or [(x[0], x[-1])]:
        inside |= (co[:, 0] > lo - 0.5) & (co[:, 0] < hi + 0.5)
    remove(kp, inside)
    kept = len(kp)
    kp.add(len(x))
    new = {"co": np.column_stack([x, y]), "type": kind, "interpolation": LINEAR if m is None else BEZIER}
    if m is not None:
        reach = np.diff(x, prepend=x[0] - (x[1] - x[0]), append=x[-1] + (x[-1] - x[-2])) / 3
        new["handle_left"] = np.column_stack([x - reach[:-1], y - m * reach[:-1]])
        new["handle_right"] = np.column_stack([x + reach[1:], y + m * reach[1:]])
        new["handle_left_type"] = new["handle_right_type"] = ALIGNED
    for attr, value in new.items():
        pair = attr in ("co", "handle_left", "handle_right")
        arr = np.zeros((len(kp), 2) if pair else len(kp), np.float32 if pair else np.int32)
        if kept:
            kp.foreach_get(attr, arr.reshape(-1))
        arr[kept:] = value
        kp.foreach_set(attr, arr.reshape(-1))
    if kept:
        kp.sort()
    fc.update()


def retype(kp, old, new):
    kind = np.zeros(len(kp), np.int32)
    kp.foreach_get("type", kind)
    kind[kind == old] = new
    kp.foreach_set("type", kind)
