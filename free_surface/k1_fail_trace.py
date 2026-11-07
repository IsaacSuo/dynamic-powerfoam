"""Where a cell's F first collapses (11-06, the random k1 failures). merged_drop.py --trace-f: logging only, nothing changes.
After each accepted step's W.end_ul, from its own inputs (still in the buffers: the trial end F sp['Fend'], the rotation
W.tRp end_ul took from it, the return factors W.tbe at the accepted z, the step's frozen C in tul['Q'], G at the accepted z
in W.tG) and its result W._Fe:
  every step   the cells whose stretch eigenvalues of sym(R^T F_end) fell below end_ul's floor 1e-6 (clamped; only the
               yielding cells go through it), the cells with det F_end <= 0, and each cell whose final F's largest singular
               value drops below 0.5 for the first time, with the record below
  watched      the same record each step from --trace-from: F0's singular values, the trial F_end's and its det, the
               eigenvalues before and after the floor, yielding or not, beta, the return's scale sc (end_ul's formula), the
               final F's singular values, L and J = exp(L), the frozen C's largest singular value, I + G(z)'s singular values
               (G at the accepted midpoint; the step's end map is I + G / theta)"""
import torch

f64 = torch.float64


def install(W, ST, step_of, watch=(7344, 6297), trace_from=200, log=print):
    end0 = W.end_ul
    seen = torch.zeros(W.n, dtype=torch.bool)
    I3c = torch.eye(3, dtype=f64)

    def rec(c, Fend, R, w, F0, Fe, be, Q, G, Ln):
        wc = w[c].clamp(min=1e-6); lw = torch.log(wc); lm = lw.mean(); dv = lw-lm
        nr = float(dv.norm().clamp(min=1e-300)); a = float(torch.exp(lm)); sc = min(1., (W.ey/a)/nr)
        sv = lambda M: torch.linalg.svdvals(M)
        return ('%d: F0 %s | F_end %s det %.3g | eig before %s after %s | %s, beta %.3f, sc %.3f | F after %s | L %.3f, J %.3f | |C| %.2f | I + G(z) %s' % (
            c, fmt(sv(F0[c])), fmt(sv(Fend[c])), float(torch.linalg.det(Fend[c])), fmt(w[c]), fmt(wc),
            'yielding' if float(be[c]) < 1 else 'elastic', float(be[c]), sc, fmt(sv(Fe[c])), float(Ln[c]), float(torch.exp(Ln[c])),
            float(sv(Q[c])[0]), fmt(sv(I3c+G[c]))))

    def end_ul(*a, **k):
        out = end0(*a, **k)
        s = step_of()
        Fend = W.tsp['Fend'].detach().cpu(); R = W.tRp.detach().cpu(); F0 = W.tsp['F0'].detach().cpu(); Fe = W._Fe.detach().cpu()
        be = W.tbe.detach().cpu(); Q = W.tul['Q'].detach().cpu(); G = W.tG.detach().cpu(); Ln = W.tul['Ln'].detach().cpu()
        A_ = R.transpose(1, 2)@Fend; w = torch.linalg.eigvalsh(.5*(A_+A_.transpose(1, 2)))
        yn = be < 1
        clamped = torch.nonzero(yn & (w[:, 0] < 1e-6)).flatten().tolist()
        inv = torch.nonzero(torch.linalg.det(Fend) <= 0).flatten().tolist()
        smax = torch.linalg.svdvals(Fe)[:, 0]
        new = torch.nonzero((smax < .5) & ~seen).flatten().tolist(); seen[new] = True
        if clamped or inv or new:
            log('  FTRACE step %d: clamped %d %s; det F_end <= 0: %d %s; first time largest sigma(F) < 0.5: %d %s' % (
                s, len(clamped), clamped[:8], len(inv), inv[:8], len(new), new[:8]), flush=True)
            for c in list(dict.fromkeys(clamped[:3]+inv[:3]+new[:3])):
                log('  FTRACE step %d cell %s' % (s, rec(c, Fend, R, w, F0, Fe, be, Q, G, Ln)), flush=True)
        if s >= trace_from:
            for c in watch:
                log('  FWATCH step %d cell %s' % (s, rec(c, Fend, R, w, F0, Fe, be, Q, G, Ln)), flush=True)
        return out
    W.end_ul = end_ul


def fmt(v):
    return '[' + ' '.join('%.3g' % float(x) for x in v) + ']'
