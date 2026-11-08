"""Exact reuse of an existing floor face for a coincident vacuum plane.

For a=-ey, b=-ground, psi_vacuum=psi_owner+2*d*(y-ground).
Inside any real power cell above ground, psi_real<=psi_owner<=psi_vacuum.
Such a vacuum generator adds no geometric constraint. Its only exposed face
is its owner's already computed floor face. Relabel that face, retaining the
full boundary state and all noncoincident vacuum constraints.
"""
import copy
import torch
from free_surface.fv_ale_planar_boundary import Planes


def split(planes,ground):
    floor=(planes.a[:,0]==0)&(planes.a[:,1]==-1)&(planes.a[:,2]==0)&(planes.b==-ground)
    keep=torch.nonzero(~floor).flatten()
    moving=Planes(*(getattr(planes,k)[keep] for k in ['owner','d','a','b']))
    return torch.nonzero(floor).flatten(),keep,moving


def attach(mesh,planes,floor,keep):
    mesh=copy.copy(mesh)
    if hasattr(mesh,'planes'):
        mesh.plane_j=keep[mesh.plane_j]
    else:
        mesh.plane_i=planes.owner.new_empty(0);mesh.plane_j=planes.owner.new_empty(0)
        mesh.plane_area=mesh.r.new_empty(0);mesh.plane_length=mesh.r.new_empty(0)
        mesh.plane_normal=mesh.p.new_empty((0,3));mesh.plane_centroid=mesh.p.new_empty((0,3))
    owners=planes.owner[floor]
    if len(torch.unique(owners))!=len(owners):raise ValueError('duplicate coincident floor seeds')
    exposed=mesh.ground_area[owners]>0;indices=floor[exposed];owners=owners[exposed]
    for key,value in [('plane_i',owners),('plane_j',indices),('plane_area',mesh.ground_area[owners]),
        ('plane_normal',mesh.ground_normal[owners]),('plane_centroid',mesh.ground_centroid[owners]),
        ('plane_length',planes.d[indices])]:
        setattr(mesh,key,torch.cat((getattr(mesh,key),value)))
    mesh.ground_area=mesh.ground_area.clone();mesh.ground_area[owners]=0.
    mesh.planes=planes.clone()
    mesh.ground_geometry_info=dict(mesh.ground_geometry_info,
        exactly_reused_floor_planes=int(len(floor)),exactly_reused_floor_faces=int(len(owners)))
    return mesh


def geometry_cpu(p,r,planes,ground):
    from free_surface.fv_ale_planar_boundary import geometry_cpu as explicit
    from free_surface.fv_ale_precise_geometry_check import cpu_mesh
    floor,keep,moving=split(planes,ground)
    mesh=explicit(p,r,moving,ground) if len(keep) else cpu_mesh(p,r,ground)
    return attach(mesh,planes,floor,keep)


def cut(solver,p,r,planes):
    from free_surface.fv_ale_planar_cutter import cut as explicit
    floor,keep,moving=split(planes,solver.ground)
    if not len(floor):return explicit(solver,p,r,planes)
    mesh=explicit(solver,p,r,moving) if len(keep) else solver.ground_cutter.geometry(p,r)
    return attach(mesh,planes,floor,keep)
