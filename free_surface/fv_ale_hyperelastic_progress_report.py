"""Plots of measured full-foam motion/energy, without rescaling trajectories."""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fv',default='hyperelastic_release_blend01_50ms')
    parser.add_argument('--prefix',default='hyperelastic_50ms')
    parser.add_argument('--label',default='FV, acoustic blend 0.1')
    parser.add_argument('--mpm',default='mpm_release_dx005_dt2p5us_50ms')
    args=parser.parse_args()
    root=Path('output/ale')
    fv=json.loads((root/(args.fv+'.json')).read_text())
    mpm=json.loads((root/(args.mpm+'.json')).read_text())
    if fv.get('stretch',1.5)!=mpm['stretch']:
        raise ValueError('FV and MPM initial strains differ')
    for key in ['E','nu']:
        if not np.isclose(fv['material'][key],mpm[key],rtol=1e-12,atol=0):
            raise ValueError('FV and MPM material '+key+' differ')
    fr=[{'time':0.,**fv['initial'],**fv['initial_spatial']}]+fv['history'];mr=mpm['rows']
    ft=np.array([r['time'] for r in fr]);mt=np.array([r['time'] for r in mr])
    fe=np.array([r['elastic_energy_joule']+r['kinetic_energy_joule'] for r in fr])
    me=np.array([r['mechanical_energy_joule'] for r in mr])
    fx=np.array([r['rms_lengths_m'] for r in fr]);mx=np.array([r['rms_lengths_m'] for r in mr])
    end=min(ft[-1],mt[-1])
    end_fe=float(np.interp(end,ft,fe));end_me=float(np.interp(end,mt,me))
    end_fx=np.array([np.interp(end,ft,fx[:,d]) for d in range(3)])
    end_mx=np.array([np.interp(end,mt,mx[:,d]) for d in range(3)])
    deficit=max(r.get('negative_remainder_deficit_joule',0.) for r in fr)
    deficit=max(deficit,fv.get('step_audit',{}).get('max_local_negative_remainder_joule',0.))
    failed_energy=deficit>1e-6*fe[0]
    local_ratio=fv.get('step_audit',{}).get('max_negative_remainder_over_local_elastic_plus_mu_over_rho0')
    local_specific=fv.get('step_audit',{}).get('max_negative_specific_remainder_joule_per_kg')
    local_warning=local_ratio is not None and local_ratio>1e-3
    note=('\nFV local deficit / (elastic + mu/rho0) reaches %.2f%%; run is under investigation.'%(100*local_ratio)
          if local_warning else ('\nFV summed negative energy remainder exceeds the audit threshold.' if failed_energy else ''))
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    ax=axes[0,0];ax.plot(ft*1000,fe/fe[0],label=args.label);ax.plot(mt*1000,me/me[0],label='PowerSim MPM, dx=5 mm')
    ax.set(xlabel='Time (ms)',ylabel='Recoverable mechanical energy / initial',ylim=(0,1.04));ax.legend(fontsize=8)
    for d,ax in enumerate([axes[0,1],axes[1,0],axes[1,1]]):
        ax.plot(ft*1000,fx[:,d]/fx[0,d],label='FV');ax.plot(mt*1000,mx[:,d]/mx[0,d],label='MPM')
        ax.set(xlabel='Time (ms)',ylabel='Mass RMS extent / initial',title='XYZ'[d]+' direction');ax.legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.25);ax.set_xlim(0,end*1000)
    fig.suptitle(f"Neo-Hookean, nu={fv['material']['nu']:.3f}: {100*(fv['stretch']-1):g}% initial prestretch, free release\n"
                 f"FV: {fv['cells']:,} geometry sites; MPM: {mpm['particles']/1e6:.2f} million particles. Neither is a converged reference."+
                 note,fontsize=11)
    fig.savefig(root/(args.prefix+'_energy_motion.png'),dpi=170);plt.close(fig)
    if (root/(args.fv+'_frames.npz')).exists() and (root/(args.mpm+'_frames.npz')).exists():
        ff=np.load(root/(args.fv+'_frames.npz'))
        mf=np.load(root/(args.mpm+'_frames.npz'))
        times=np.linspace(0,end,4);fig,axes=plt.subplots(2,4,figsize=(12,6),layout='constrained')
        for column,t in enumerate(times):
            k=int(np.abs(ff['t']-t).argmin());p=ff['p'][k];r=ff['r'][k]
            ax=axes[0,column]
            circles=EllipseCollection(2*r,2*r,np.zeros_like(r),units='xy',offsets=p[:,:2],offset_transform=ax.transData,
                                      facecolors='#287cb2',edgecolors='none')
            ax.add_collection(circles);ax.set_title(f"FV: {ff['t'][k]*1000:.1f} ms")
            k=int(np.abs(mf['t']-t).argmin());x=mf['x'][k]
            axes[1,column].scatter(x[:,0],x[:,1],s=.2,c='#dc8838',linewidths=0,rasterized=True)
            axes[1,column].set_title(f"MPM: {mf['t'][k]*1000:.1f} ms")
        for ax in axes.flat:
            ax.set(xlim=(.065,.765),ylim=(-.20,.50),aspect='equal');ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle('Orthographic XY silhouettes; fixed scale and camera\nFV: union of actual balls. MPM: sampled material points.',fontsize=11)
        fig.savefig(root/(args.prefix+'_shapes.png'),dpi=170);plt.close(fig)
    shared=mt<=end+1e-14
    interp=np.stack([np.interp(mt[shared],ft,fx[:,d]) for d in range(3)],1)
    result={'fv_completed':fv['completed'],'mpm_status':mpm['status'],
            'max_observed_local_negative_remainder_deficit_J':deficit,
            'negative_remainder_exceeds_one_ppm_initial_energy':bool(failed_energy),
            'max_step_local_negative_specific_energy_J_per_kg':local_specific,
            'max_step_local_negative_over_elastic_plus_mu_over_rho0':local_ratio,
            'local_specific_deficit_exceeds_diagnostic_1e3rd_scale':bool(local_warning),
            'every_step_specific_deficit_recorded':local_ratio is not None,
            'comparison_end_time_seconds':float(end),
            'fv_mechanical_loss_fraction':float(1-end_fe/fe[0]),'mpm_mechanical_loss_fraction':float(1-end_me/me[0]),
            'end_rms_extent_relative_difference':((end_fx-end_mx)/end_mx).tolist(),
            'max_rms_extent_absolute_difference_m':np.abs(interp-mx[shared]).max(0).tolist(),
            'fv_center_drift_m':(np.array(fr[-1]['com_world_m'])-np.array(fr[0]['com_world_m'])).tolist(),
            'fv_total_momentum_change_kg_m_s':fv.get('total_momentum_change_kg_m_s'),
            'fv_angular_momentum_change_kg_m2_s':(np.array(fr[-1]['angular_momentum_about_com_kg_m2_s'])-
                np.array(fr[0]['angular_momentum_about_com_kg_m2_s'])).tolist()
                if 'angular_momentum_about_com_kg_m2_s' in fr[0] else None,
            'local_energy_audit_scope':fv.get('step_audit',{}).get('scope','output records only'),
            'fv_time_geometry_work_error_J':fv.get('energy_ledger',{}).get('step_time_geometry_error_joule'),
            'fv_actual_mean_step_seconds':fv['time_reached']/max(fv['steps'],1),
            'fv_diagnostic_wall_seconds':fv['seconds'],
            'caveat':'Different spatial resolutions and FP64/FP32; MPM is a comparison, not converged truth.'}
    (root/(args.prefix+'_comparison.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
