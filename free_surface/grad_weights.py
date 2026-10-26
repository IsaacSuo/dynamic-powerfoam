"""Cell gradient with bond-midpoint weights (2026-10-22). The clean model's F_B is the weighted least-squares fit of
dx_j = F dX_j over the cell's bonds with weight A h_B / L^2 (h_B the site-to-face distance: the face moment A h n n of the
pyramid from the site), walls with h_B <= 0 left out. A site close to its own faces gives them ~0 weight; a site at a
corner of its cell (output/pacnerf_dem_cutft, body 4618: 3 of 5 faces at h = -0.05..-0.12 mm) leaves F_B singular.
Here every bond weighs A / (2 L) on both sides (moment A (L / 2) n n: the face put at the bond midpoint), always > 0 and
the same for the two cells; the air faces likewise A (L_air / 2) along their lever, exposed sphere patches as before.
Still exact for any affine motion (least squares). Only the gradient changes: the volume (signed pyramids, Hermite wall
positions) and the completion springs keep the true h."""
import torch

from prototypes.kb_derivation.clean_model import CleanTerms


class MidpointTerms(CleanTerms):
    def __init__(self, model, sc, geometry, variance=True):
        super().__init__(model, sc, geometry, variance)
        n, dt, dev = self.n, self.X.dtype, self.X.device
        nn = self.nrm[:, :, None]*self.nrm[:, None, :]
        half = (self.A*self.L/2)[:, None, None]*nn
        self.M = torch.zeros(n, 3, 3, dtype=dt, device=dev).index_add(0, self.i, half).index_add(0, self.j, half)
        nair = len(sc.air_faces['body'])
        lev = sc.vs_lever; Ls = lev.norm(dim=1)
        area = torch.as_tensor(sc.air_faces['area'], dtype=dt, device=dev)
        w = torch.cat((area/(2*Ls[:nair]), sc.vs_k[nair:]/(2*float(model.mu_ref))))       # times l l^T: A (L / 2) along the lever
        self.Mair = torch.zeros(n, 3, 3, dtype=dt, device=dev).index_add(0, sc.vs_body, w[:, None, None]*lev[:, :, None]*lev[:, None, :])
        self.MN_inv = torch.linalg.inv(self.nbr_sum(self.M))
        self.Mt_inv = torch.linalg.inv(self.M+self.Mair)

    def F(self, x):
        d = x[self.j]-x[self.i]
        t = (self.A/2)[:, None, None]*d[:, :, None]*self.nrm[:, None, :]
        S = torch.zeros(self.n, 3, 3, dtype=x.dtype, device=x.device).index_add(0, self.i, t).index_add(0, self.j, t)
        FN = self.nbr_sum(S)@self.MN_inv
        return (S+FN@self.Mair)@self.Mt_inv
