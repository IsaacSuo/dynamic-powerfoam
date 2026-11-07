"""ALE geometry/transport adapter on the actual sphere-clipped power cells.

This module does NOT replace the constitutive law or integrate solid dynamics.
It supplies exact instantaneous mesh-volume fluxes and first-order conservative
transport for the remap phase. Positive wall flux is outward from i towards j.
No cell mass is reset from its new volume, and no target volume/radius fit is used.

For y on a shared radical plane:
  w.n = ((y-p_i).p_dot_i - (y-p_j).p_dot_j
         + r_i*r_dot_i - r_j*r_dot_j) / |p_j-p_i|.
On the exposed sphere, w.n = p_dot_i.n + r_dot_i.
Integration is exact using the existing wall centroid, cap area and cap normal
integral. These are Reynolds-transport geometry identities, not a solid model.
"""
from dataclasses import dataclass
import torch


@dataclass
class PowerControlVolumes:
    p: torch.Tensor
    r: torch.Tensor
    volume: torch.Tensor
    i: torch.Tensor
    j: torch.Tensor
    area: torch.Tensor
    normal: torch.Tensor
    centroid: torch.Tensor
    cap_area: torch.Tensor
    cap_normal_integral: torch.Tensor

    @classmethod
    def from_geometry(cls, g):
        # Clone: the production GeoPipe reuses its output buffers on the next cut.
        keep = g.wall_A > 0
        return cls(g.p.clone(), g.r.clone(), g.V.clone(), g.i[keep].clone(),
                   g.j[keep].clone(), g.wall_A[keep].clone(), g.nn[keep].clone(),
                   g.wall_c[keep].clone(), g.patch_A.clone(), g.patch_n.clone())

    def incidence(self, wall):
        shape = (len(self.p),) + wall.shape[1:]
        return torch.zeros(shape, dtype=wall.dtype, device=wall.device).index_add(
            0, self.i, wall).index_add(0, self.j, -wall)

    def mesh_flux(self, site_velocity, radius_velocity):
        if site_velocity.shape != self.p.shape or radius_velocity.shape != self.r.shape:
            raise ValueError("mesh velocities must have shapes (cells, 3) and (cells,)")
        i, j, c = self.i, self.j, self.centroid
        length = (self.p[j]-self.p[i]).norm(dim=1)
        wall = self.area / length * (
            ((c-self.p[i])*site_velocity[i]).sum(1)
            - ((c-self.p[j])*site_velocity[j]).sum(1)
            + self.r[i]*radius_velocity[i]-self.r[j]*radius_velocity[j])
        cap = (self.cap_normal_integral*site_velocity).sum(1) + self.cap_area*radius_velocity
        return wall, cap

    def volume_rate(self, site_velocity, radius_velocity):
        wall, cap = self.mesh_flux(site_velocity, radius_velocity)
        return self.incidence(wall)+cap

    def affine_material_flux(self, gradient, offset):
        """Integral of u.n for u(y)=offset+gradient@y, including curved caps.

        Obtain int n n^T on caps from the first moment divergence identity:
        sum_walls int (y-p_i) tensor n + r_i int_cap n tensor n = V_i I.
        No fitted G and no cap flattening are used.
        """
        i, j, c, n = self.i, self.j, self.centroid, self.normal
        outer = lambda a, b: a[:, :, None]*b[:, None, :]
        wall_moment = torch.zeros((len(self.p),3,3), dtype=self.p.dtype, device=self.p.device)
        wall_moment.index_add_(0,i,self.area[:,None,None]*outer(c-self.p[i],n))
        wall_moment.index_add_(0,j,-self.area[:,None,None]*outer(c-self.p[j],n))
        eye = torch.eye(3,dtype=self.p.dtype,device=self.p.device)
        cap_nn = (self.volume[:,None,None]*eye-wall_moment)/self.r[:,None,None]
        wall = self.area*((offset+c@gradient.T)*n).sum(1)
        cap = ((offset+self.p@gradient.T)*self.cap_normal_integral).sum(1)
        cap += self.r*(cap_nn*gradient[None]).sum((1,2))
        return wall, cap

    def remap_rate(self, extensive, site_velocity, radius_velocity):
        """Pure mesh motion through stationary material, closed fixed exterior.

        Donor-cell FV discretization of d int_cell U / dt = int_boundary U w.n.
        The caller must explicitly handle any moving exterior; silently treating
        an exposed cap as an internal face would create/destroy material.
        """
        wall, cap = self.mesh_flux(site_velocity,radius_velocity)
        scale = wall.abs().sum().clamp(min=1e-30)
        if float(cap.abs().sum()/scale) > 1e-10:
            raise ValueError('remap needs exterior boundary transport; caps are moving: relative=%g cell=%d area=%g' % (float(cap.abs().sum()/scale),int(cap.abs().argmax()),float(self.cap_area[cap.abs().argmax()])))
        density = extensive/self.volume[:,None]
        # w.n>0: i grows into j, so j donates to i.
        donor = torch.where(wall[:,None]>=0,density[self.j],density[self.i])
        return self.incidence(wall[:,None]*donor)

    def remap_cfl(self, site_velocity, radius_velocity):
        wall, cap = self.mesh_flux(site_velocity,radius_velocity)
        outgoing = torch.zeros_like(self.volume)
        outgoing.index_add_(0,self.i,(-wall).clamp(min=0))
        outgoing.index_add_(0,self.j,wall.clamp(min=0))
        return float((self.volume/outgoing.clamp(min=1e-300)).min())
