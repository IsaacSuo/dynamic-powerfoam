"""The full state of a k1 run between two steps (11-06, for replaying a failing step). Logging/replay only.

take(roots) walks the objects' attributes and their dicts (roots: {'W': MergedUL, 'ST': MergedStep, 'CG': TrustCG}) and
copies every Warp array, torch tensor, numpy array, scalar, and tuple/list of these, by path. Other objects (the recut,
geometry, graphs, kernels, functions, back references) are not walked; inventory() lists what was skipped. restore() puts
each value back by path: arrays and tensors of the same shape are copied in place (the captured CUDA graphs keep their
pointers), the rest is rebound. A snapshot lives on the device (each step, cheap) or on the host (to save to a file)."""
import numpy as np
import torch
import warp as wp

LEAF = (int, float, bool, str, type(None), np.generic)


def _items(obj):
    if isinstance(obj, dict):
        return [(('k', k), v) for k, v in obj.items()]
    return [(('a', k), v) for k, v in vars(obj).items()]


def _get(c, key):
    return c[key[1]] if key[0] == 'k' else getattr(c, key[1])


def _set(c, key, v):
    if key[0] == 'k':
        c[key[1]] = v
    else:
        setattr(c, key[1], v)


def _copy_val(v, host):
    if isinstance(v, wp.array):
        try:
            t = wp.to_torch(v, requires_grad=False)
        except Exception:                                                           # (a Warp type torch has no view for)
            return ('wpnp', v.numpy().copy())
        return ('wp', t.detach().cpu().clone() if host else t.detach().clone())
    if isinstance(v, torch.Tensor):
        return ('t', v.detach().cpu().clone() if host else v.detach().clone())
    if isinstance(v, np.ndarray):
        return ('np', v.copy())
    if isinstance(v, LEAF):
        return ('s', v)
    if isinstance(v, (tuple, list)) and all(isinstance(x, LEAF+(torch.Tensor, tuple, list)) for x in v):
        return ('seq', type(v).__name__, [_copy_val(x, host) for x in v])
    return None


def take(roots, host=False, skip=()):
    snap, skipped, seen = {}, [], set()

    def walk(obj, path):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        for key, v in _items(obj):
            p = path+(key,)
            if '.'.join(str(k[1]) for k in p) in skip:
                continue
            if isinstance(v, dict):
                walk(v, p); continue
            c = _copy_val(v, host)
            if c is None:
                skipped.append(('.'.join(str(k[1]) for k in p), type(v).__name__)); continue
            snap[p] = c
    for name, obj in roots.items():
        walk(obj, (('r', name),))
    return dict(values=snap, skipped=skipped)


def _restore_val(cur, saved, dev):
    kind = saved[0]
    if kind == 'wp':
        if isinstance(cur, wp.array):
            t = wp.to_torch(cur, requires_grad=False)
            if tuple(t.shape) == tuple(saved[1].shape):
                t.copy_(saved[1].to(t.device)); return cur
        raise RuntimeError('restore: a Warp array changed shape or type')
    if kind == 'wpnp':
        if isinstance(cur, wp.array) and tuple(cur.shape) == tuple(saved[1].shape[:len(cur.shape)]):
            wp.copy(cur, wp.array(saved[1], dtype=cur.dtype, device=cur.device)); return cur
        raise RuntimeError('restore: a Warp array changed shape or type')
    if kind == 't':
        if isinstance(cur, torch.Tensor) and tuple(cur.shape) == tuple(saved[1].shape) and cur.dtype == saved[1].dtype:
            cur.copy_(saved[1].to(cur.device)); return cur
        return saved[1].to(dev).clone()
    if kind == 'np':
        return saved[1].copy()
    if kind == 's':
        return saved[1]
    if kind == 'seq':
        vals = [_restore_val(None, x, dev) for x in saved[2]]
        return tuple(vals) if saved[1] == 'tuple' else vals
    raise RuntimeError(kind)


def restore(roots, snap, dev='cuda'):
    for p, saved in snap['values'].items():
        c = roots[p[0][1]]
        for key in p[1:-1]:
            try:
                c = _get(c, key)
            except (KeyError, AttributeError):                                    # (a dict the snapshot walked that is missing now, e.g. a step's stats)
                _set(c, key, {}); c = _get(c, key)
        key = p[-1]
        try:
            cur = _get(c, key)
        except (KeyError, AttributeError):
            cur = None
        new = _restore_val(cur, saved, dev)
        if new is not cur:
            _set(c, key, new)


def to_host(snap):
    h = {}
    for p, v in snap['values'].items():
        h[p] = _to_host(v)
    return dict(values=h, skipped=snap['skipped'])


def _to_host(v):
    if v[0] in ('wp', 't'):
        return (v[0], v[1].cpu())
    if v[0] == 'seq':
        return ('seq', v[1], [_to_host(x) for x in v[2]])
    return v


def inventory(snap):
    nb = 0
    for v in snap['values'].values():
        if v[0] in ('wp', 't'):
            nb += v[1].numel()*v[1].element_size()
    return '%d values, %.1f MB in arrays; not walked: %s' % (len(snap['values']), nb/2**20, ', '.join('%s (%s)' % s for s in snap['skipped']))
