"""why the cap traction run stopped at the first contacts: per step from 225, the largest traction strain |t| / (lam + 2 mu)
and its cell's cap area against the median cap, the ball's overlap and force, and the closure's det F_B there"""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import free_surface.lion_inverse as li
li.set_scene('pacnerf_e0'); li.GLUE = 'material'; li.CAPT = 1


def probe(s, W, ST, cbody, Rb):
    if s < 225:
        return
    t = W.ttrac.norm(dim=1)*float(W.ttpar[0])
    k = int(torch.argmax(t)); A = W.cap_area
    medA = float(A[W.st.cap].median())
    bi = torch.nonzero(cbody == k).flatten()
    fn = float(W.tbfn[bi].sum()) if len(bi) else 0.; de = float(W.tbdelta[bi].max()) if len(bi) else 0.
    print('step %d: max traction strain %.3g at cell %d (cap area / median %.3g, R %.4f, overlap %.2e, F_n %.3g); det F_B there %.3f; touching %d'
          % (s, float(t[k]), k, float(A[k])/medA, float(W.st.r[k]), de, fn, float(torch.linalg.det(W.tF[k])), int((W.tbfn > 0).sum())), flush=True)


try:
    li.forward(np.array([10.04361, .30079413, -5.809143]), 5, verbose=False, on_step=probe)
except RuntimeError as e:
    print('stopped:', e)
