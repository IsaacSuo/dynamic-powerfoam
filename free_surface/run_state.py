"""a run's whole state to a file and back, for the checks that need the real one (ring_yield_drop.py's)"""
import torch
import warp as wp

dev = 'cuda'


def wp_put(a, v):
    t_ = wp.to_torch(a, requires_grad=False)
    t_.copy_(torch.as_tensor(v, device=t_.device, dtype=t_.dtype))



def pack(o):
    out = {}
    for k_, v_ in o.__dict__.items():
        if isinstance(v_, (bool, int, float, str, type(None))):
            out[k_] = v_
        elif isinstance(v_, torch.Tensor):
            out[k_] = v_.detach().cpu().clone()
        elif isinstance(v_, wp.array):
            out[k_] = ('wp', v_.numpy().copy())
        elif isinstance(v_, (tuple, list)) and all(isinstance(a__, (torch.Tensor, int, float, type(None))) for a__ in v_):
            out[k_] = (type(v_).__name__, [a__.detach().cpu().clone() if isinstance(a__, torch.Tensor) else a__ for a__ in v_])
        elif isinstance(v_, dict):
            d_ = {}
            for k2, v2 in v_.items():
                if isinstance(v2, wp.array):
                    d_[k2] = ('wp', v2.numpy().copy())
                elif isinstance(v2, torch.Tensor):
                    d_[k2] = v2.detach().cpu().clone()
                elif isinstance(v2, (bool, int, float, str, type(None))):
                    d_[k2] = v2
            out[k_] = ('dict', d_)
    return out


def wp_put(a, v):
    t_ = wp.to_torch(a, requires_grad=False)
    v_ = torch.as_tensor(v, device=t_.device, dtype=t_.dtype)
    if v_.shape != t_.shape and v_.shape[1:] == t_.shape[1:]:                       # (a buffer of another capacity, as the contact list: the common part, the rest zero)
        print('unpack: an array of %d rows into one of %d' % (v_.shape[0], t_.shape[0]), flush=True)
        m_ = min(v_.shape[0], t_.shape[0])
        t_.zero_(); t_[:m_].copy_(v_[:m_])
        return
    t_.copy_(v_)


def unpack(o, st):
    for k_, v_ in st.items():
        cur = getattr(o, k_, None)
        if isinstance(v_, torch.Tensor):
            if isinstance(cur, torch.Tensor) and cur.shape == v_.shape:
                cur.copy_(v_.to(cur.device))
            else:
                setattr(o, k_, v_.to(dev))
        elif isinstance(v_, tuple) and v_[0] == 'wp':
            if isinstance(cur, wp.array):
                wp_put(cur, v_[1])
        elif isinstance(v_, tuple) and v_[0] == 'dict':
            if isinstance(cur, dict):
                for k2, v2 in v_[1].items():
                    c2 = cur.get(k2)
                    if isinstance(v2, tuple) and v2[0] == 'wp':
                        if isinstance(c2, wp.array):
                            wp_put(c2, v2[1])
                    elif isinstance(v2, torch.Tensor):
                        if isinstance(c2, torch.Tensor) and c2.shape == v2.shape:
                            c2.copy_(v2.to(c2.device))
                        else:
                            cur[k2] = v2.to(dev)
                    else:
                        cur[k2] = v2
            else:
                setattr(o, k_, {k2: (v2.to(dev) if isinstance(v2, torch.Tensor) else v2) for k2, v2 in v_[1].items()})
        elif isinstance(v_, tuple) and v_[0] in ('tuple', 'list'):
            a__ = [b_.to(dev) if isinstance(b_, torch.Tensor) else b_ for b_ in v_[1]]
            setattr(o, k_, tuple(a__) if v_[0] == 'tuple' else a__)
        else:
            setattr(o, k_, v_)


