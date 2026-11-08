"""Read-only decomposition of the frozen free-state wave bound."""
import json
from pathlib import Path
import torch,warp as wp
from free_surface.fv_ale_cfl_diagnose import load
from free_surface.fv_ale_solid import State

def wave(sol,s,suppress_surface=()):
    m=s.mesh;g=sol.groups;i,j,A,n,c=m.i,m.j,m.area,m.normal,m.centroid
    b={'q':s.q,'V':m.volume,'groups':g,'p':s.p,'r':s.r,'i':i,'j':j,'A':A,'n':n,'c':c,
       'Q':m.cap_Q,'N':m.cap_normal_integral,'cap':m.cap_area,'surface':m.surface,'floor':m.ground_area}
    ops=sol.device_ops();b.update(ops.extra_rhs_values(m))
    if suppress_surface:
        b['surface']=b['surface'].clone();b['surface'][list(suppress_surface)]=False
    total,V,v,rho,p,stress,sigma,sound,_=ops.primitive(s.q,m.volume,g)
    w,dr,_,_,_,us,*_=ops.mechanical_kinematics(b,v,sigma,rho*sound)
    w,dr=ops.mesh_motion(b,w,dr,us)
    wall,cap=m.mesh_flux(w,dr)
    rel=A*(us*n).sum(1)-wall;ext=g[i]!=g[j];rel=torch.where(ext,rel,0.)
    outgoing=torch.zeros_like(m.volume).index_add(0,i,rel.clamp(min=0)).index_add(0,j,(-rel).clamp(min=0))
    area=m.cap_area.clamp(min=0)+m.ground_area+torch.zeros_like(m.volume).index_add(0,i,A*ext).index_add(0,j,A*ext)
    W=sol.group_sum(area*sound)[g];O=sol.group_sum(outgoing)[g];dt=V/(W+O).clamp(min=1e-300)
    result={'min_dt_us':float(dt.min())*1e6,'wave_cell':int(dt.argmin()),'groups':{}}
    for root in [4211,7191,4337]:
        members=torch.where(g==root)[0];touch=(g[i]==root)|(g[j]==root)
        edges=torch.where(touch&ext)[0];flux=torch.where(g[i][edges]==root,rel[edges],-rel[edges]);order=flux.argsort(descending=True)[:5]
        result['groups'][root]={'volume':float(V[root]),'sound':float(sound[root]),'area':float(sol.group_sum(area)[root]),
            'wave_denominator':float(W[root]),'outflow_denominator':float(O[root]),'dt_us':float(dt[root])*1e6,
            'members':members.tolist(),'surface_area':float(m.cap_area[members].sum()),
            'member_motion':[{'id':int(k),'surface':bool(b['surface'][k]),'cap_area':float(m.cap_area[k]),
                'v':v[k].tolist(),'w':w[k].tolist(),'dr':float(dr[k]),'radius':float(s.r[k]),
                'Q_over_cap':(m.cap_Q[k]/m.cap_area[k].clamp(min=1e-300)).tolist() if m.cap_area[k]>0 else None,
                'G_norm':float(ops._compatible['gradient'][k].norm())} for k in members],
            'top_outflow':[{'i':int(i[e]),'j':int(j[e]),'area':float(A[e]),'relative_flux':float(flux[k]),
                           'fluid_normal_speed':float((us[e]*n[e]).sum()),'wall_flux':float(wall[e])} for k,e in zip(order,edges[order])]}
    return result

@torch.no_grad()
def main():
    torch.set_num_threads(1);wp.init();sol,s,tol=load(Path('output/ale/cfl2_entry/fv_state_cfl05_free5ms.pt'),.01)
    first=sol.rhs(s);rows=[]
    for h in [0.,2e-8,5e-8,1e-7,2e-7,1e-6,1e-5,4e-5]:
        p,r,q=sol.device_ops().update(s,h,first);m=sol.cut(p,r);q=sol.plastic_return(q,m);stage=State(p,r,q,m)
        rhs=sol.rhs(stage);d=wave(sol,stage);d['dt']=h;d['production_wave_dt_us']=rhs[3]['wave_dt_scale']*1e6
        assert abs(d['min_dt_us']-d['production_wave_dt_us'])<1e-8
        rows.append(d)
        if h==2e-7:
            changed=wave(sol,stage,suppress_surface=[4987,7191]);changed.update(dt=h,diagnostic_only='suppress new cap surface flag on 4987/7191')
            rows.append(changed)
    Path('output/ale/cfl_wave_probe.json').write_text(json.dumps(rows,indent=2)+'\n')
    for r in rows:print(json.dumps(r),flush=True)

if __name__=='__main__':main()
