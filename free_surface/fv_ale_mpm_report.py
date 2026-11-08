"""Summarize matched MPM/FV release; never label MPM as exact ground truth."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('output/ale')
    fv=json.loads((root/'hyperelastic_release_roundoff.json').read_text())
    names=['mpm_release_dx020_dt20','mpm_release_dx020_dt10','mpm_release_dx010_dt10',
           'mpm_release_dx010_dt5','mpm_release_dx005_dt5','mpm_release_dx005_dt2p5']
    runs={name:json.loads((root/(name+'.json')).read_text()) for name in names}
    result={'comparison':'same 50% prestrain release, same neo-Hookean energy, no gravity/contact/damping',
            'fv':{'initial_elastic_joule':fv['initial']['elastic_energy_joule'],
                  'final':fv['final'],'steps':fv['steps'],'seconds':fv['seconds']},'mpm':{}}
    for name,d in runs.items():
        r=d['rows'][-1]
        result['mpm'][name]={k:d[k] for k in ['dx','dt','particles','steps','status','raw_sampling_volume_relative_error',
                                                'mass_relative_error','initial_stress_relative_error','seconds_including_first_kernel_compilation']}
        result['mpm'][name].update(initial_elastic_joule=d['rows'][0]['elastic_energy_joule'],
                                  initial_energy_relative_difference_to_fv=d['rows'][0]['elastic_energy_joule']/fv['initial']['elastic_energy_joule']-1,
                                  final=r)
    result['time_refinement']={}
    for a,b in zip(names[::2],names[1::2]):
        x=runs[a]['rows'][-1];y=runs[b]['rows'][-1]
        result['time_refinement'][str(runs[a]['dx'])]={
            'loss_percentage_point_difference':100*(y['mechanical_loss_fraction']-x['mechanical_loss_fraction']),
            'end_rms_length_difference_m':(np.array(y['rms_lengths_m'])-np.array(x['rms_lengths_m'])).tolist()}
    # Plot particle translational mechanical energy; display APIC affine norm
    # separately, since it is not identical to continuum resolved kinetic energy.
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),constrained_layout=True)
    initial=fv['initial']['elastic_energy_joule']
    hist=[{'time':0.,**fv['initial']}]+fv['history']
    axes[0].plot([h['time']*1000 for h in hist],
        [100*(h['elastic_energy_joule']+h['kinetic_energy_joule'])/initial for h in hist],label='FV (7,690 geometric cells)',color='#c94646',lw=2.5)
    selected=[names[1],names[3],names[5]]
    colors=['#87bddd','#3b87b3','#146141']
    for name,color in zip(selected,colors):
        d=runs[name];rows=d['rows'];E=rows[0]['mechanical_energy_joule']
        axes[0].plot([h['time']*1000 for h in rows],[100*h['mechanical_energy_joule']/E for h in rows],
                     label=f'MPM {d["dx"]*1000:g} mm grid',color=color,lw=2)
    d=runs[selected[-1]];E=d['rows'][0]['mechanical_energy_joule']
    axes[0].plot([h['time']*1000 for h in d['rows']], [100*h['mechanical_plus_affine_joule']/E for h in d['rows']],
                 '--',color=colors[-1],lw=1,label='5 mm MPM + APIC affine energy')
    axes[0].set(xlabel='Time (ms)',ylabel='Recoverable elastic + kinetic energy (%)',ylim=(0,103),xlim=(0,10))
    axes[0].grid(alpha=.2);axes[0].legend(fontsize=8,loc='lower left')
    vals=[[fv['final']['elastic_energy_joule']/initial,fv['final']['kinetic_energy_joule']/initial,0,fv['final']['nonelastic_remainder_joule']/initial]]
    for name in selected:
        rows=runs[name]['rows'];r=rows[-1];E=rows[0]['mechanical_energy_joule']
        vals.append([r['elastic_energy_joule']/E,r['kinetic_energy_joule']/E,r['apic_affine_kinetic_joule']/E,r['mechanical_plus_affine_loss_fraction']])
    vals=np.array(vals)*100;bottom=np.zeros(len(vals))
    for j,(label,color) in enumerate(zip(['Elastic','Kinetic','APIC affine','Remainder / untracked loss'],['#3479b4','#68ad90','#d4b15e','#d87570'])):
        axes[1].bar(np.arange(4),vals[:,j],bottom=bottom,label=label,color=color,width=.65);bottom+=vals[:,j]
    axes[1].set(xticks=np.arange(4),xticklabels=['FV','MPM\n20 mm','MPM\n10 mm','MPM\n5 mm'],ylabel='Initial energy (%) at 10 ms',ylim=(0,103))
    axes[1].legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,-.14),ncol=2)
    fig.suptitle('Matched hyperelastic release: same material and prestrain; MPM is a numerical reference')
    fig.savefig(root/'mpm_fv_energy_comparison.png',dpi=180)
    fig.savefig(root/'mpm_fv_energy_comparison.svg')
    (root/'mpm_fv_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,d in runs.items():
        row=d['rows'][-1]
        print(name,'N',d['particles'],'dt_us',d['dt']*1e6,'loss%',100*row['mechanical_loss_fraction'],
              'including_affine_loss%',100*row['mechanical_plus_affine_loss_fraction'],
              'seconds',d['seconds_including_first_kernel_compilation'])
    print('dt convergence',json.dumps(result['time_refinement']))


if __name__=='__main__':main()
