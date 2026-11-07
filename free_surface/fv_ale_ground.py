"""Fixed planar wall traction for directly half-space-clipped power cells.

The geometry adapter constructs an actual floor face. Curved cap quadrature
and penalty contact are not used in this mode. Unloading/release remains an
explicit unsupported condition, rather than silently applying adhesion.
"""
import torch
from free_surface.fv_ale_ground_geometry import GroundClip



def plane_boundary(mesh,v,sigma,Z,friction):
    """Stationary impermeable plane, compressive acoustic reaction and Coulomb.

    With n=-ey, t=sigma*n+Z*(u-v), u_y=0 gives P=-sigma_yy-Z*v_y.
    Do not silently turn a tensile prediction into adhesive contact. A tensile
    reaction is reported; initial compression tests must establish its absence.
    Static/kinetic Coulomb follows the tangential acoustic relation exactly.
    """
    area=mesh.ground_area;active=area>0
    tn=-sigma[:,:,1];free=v-tn/Z[:,None]
    raw_pressure=-Z*free[:,1]
    pressure=raw_pressure.clamp(min=0)*active
    tangent=free[:,[0,2]];speed=tangent.norm(dim=1)
    amount=torch.minimum(friction*pressure,Z*speed)
    traction=torch.zeros_like(v);traction[:,1]=pressure
    traction[:,[0,2]]=-amount[:,None]*tangent/speed.clamp(min=1e-300)[:,None]
    velocity=free+traction/Z[:,None];velocity[:,1]=0.
    velocity=torch.where(active[:,None],velocity,v)
    force=area[:,None]*traction
    heat=-area*(traction[:,[0,2]]*velocity[:,[0,2]]).sum(1)
    # A stationary rigid floor does no normal work. Tangential mechanical loss
    # is retained as material heat, so the total-energy boundary source is zero.
    tensile=active&(raw_pressure < -1e-8*(Z*v.norm(dim=1)+sigma.norm(dim=(1,2))+1.))
    return force,velocity,heat,{
        'contact_cells':int(active.sum()),'contact_area':float(area.sum()),
        'wall_pressure_min':float(raw_pressure[active].min()) if bool(active.any()) else 0.,
        'wall_pressure_max':float(pressure.max()),'wall_tensile_prediction_cells':int(tensile.sum()),
        'wall_tensile_prediction_area':float(area[tensile].sum()),
        'wall_sliding_cells':int((active&(Z*speed>friction*pressure)).sum()),
        'wall_normal_velocity_max':float(velocity[active,1].abs().max()) if bool(active.any()) else 0.}
