"""Measured impact/rebound history; separate completed horizon from separation."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);args=ap.parse_args()
    root=Path('output/ale');data=json.loads((root/(args.tag+'.json')).read_text())
    rows=[{'time':0.,**data['initial'],**data['initial_spatial']}]+data['history']
    t=np.array([r['time'] for r in rows]);velocity=np.array([r['com_velocity_m_s'] for r in rows])
    area=np.array([r['contact_area_m2'] for r in rows])
    kinetic=np.array([r['kinetic_energy_joule'] for r in rows]);elastic=np.array([r['elastic_energy_joule'] for r in rows])
    remainder=np.array([r['nonelastic_remainder_joule'] for r in rows])
    recent=t>=t[-1]-.005-1e-12
    separated=bool(recent.sum()>=5 and np.all(area[recent]<1e-10) and np.all(velocity[recent,1]>0))
    figure,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    ax[0,0].plot(t*1000,kinetic,label='Kinetic');ax[0,0].plot(t*1000,elastic,label='Elastic')
    ax[0,0].plot(t*1000,remainder,label='Nonelastic remainder');ax[0,0].set_ylabel('Energy (J)');ax[0,0].legend()
    ax[0,1].plot(t*1000,velocity[:,1]);ax[0,1].axhline(0,color='black',lw=.7);ax[0,1].set_ylabel('Vertical COM velocity (m/s)')
    ax[1,0].plot(t*1000,area*1e4);ax[1,0].set_ylabel('Ground contact area (cm²)')
    ax[1,1].plot(t*1000,[r['J_min'] for r in rows],label='Minimum')
    ax[1,1].plot(t*1000,[r['J_max'] for r in rows],label='Maximum');ax[1,1].set_ylabel('Material volume ratio J');ax[1,1].legend()
    for panel in ax.flat:panel.set_xlabel('Time (ms)');panel.grid(alpha=.25)
    figure.suptitle('Full power-ball Neo-Hookean impact; frictionless ground, no gravity\n'+
        ('Upward, contact-free for the final 5 ms' if separated else 'Complete separation not yet established'))
    figure.savefig(root/(args.tag+'_contact_history.png'),dpi=160);plt.close(figure)
    result={'completed_requested_horizon':data['completed'],'time_reached':data['time_reached'],
        'five_ms_sustained_separation':separated,'final_vertical_velocity_m_s':float(velocity[-1,1]),
        'minimum_sampled_vertical_velocity_increment':float(np.diff(velocity[:,1]).min()),
        'max_horizontal_com_velocity_m_s':float(np.abs(velocity[:,[0,2]]).max()),
        'final_contact_area_m2':float(area[-1]),'max_contact_area_m2':float(area.max()),
        'mechanical_loss_fraction':float(1-(kinetic[-1]+elastic[-1])/(kinetic[0]+elastic[0])),
        'max_recorded_total_energy_error_joule':float(np.max(np.abs(kinetic+elastic+remainder-(kinetic[0]+elastic[0]+remainder[0])))),
        'step_audit':data['step_audit'],'steps':data['steps'],'cumulative_wall_seconds':data['seconds'],
        'boundary_insertions':len(data.get('boundary_insertions',[])),
        'boundary_prunings':len(data.get('boundary_prunings',[])),
        'boundary_insertion_energy_roundoff_joule':sum(r.get('mechanical_energy_roundoff_change_joule',0) for r in data.get('boundary_insertions',[])),
        'boundary_pruning_energy_roundoff_joule':sum(r.get('mechanical_energy_roundoff_change_joule',0) for r in data.get('boundary_prunings',[])),
        'limits':'Full sampled trajectory and per-step energy audit; no spatial-convergence claim and no friction support.'}
    (root/(args.tag+'_contact_summary.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
