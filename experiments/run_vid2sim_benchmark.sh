#!/bin/bash
# Vid2Sim GSO benchmark (12 objects, material identification) on the current pipeline (springgaus_train.py, blend render).
#
#   bash experiments/run_vid2sim_benchmark.sh setup [obj ...]          static foam + DEM + trainer data (default: all 12)
#   bash experiments/run_vid2sim_benchmark.sh check [obj ...]          0-iteration check: frame-0 / free-fall IoU per object
#   bash experiments/run_vid2sim_benchmark.sh start_test               sofa from E = 2e4, 1e5, 1e6 (nu 0.3): start sensitivity
#   bash experiments/run_vid2sim_benchmark.sh all [E0] [obj ...]       every object from E0 (default 1e5), nu 0.3
#   bash experiments/run_vid2sim_benchmark.sh summary <tag>            table of output/vid2sim_eval/<tag>/<obj>/report.json
#
# Needs data/vid2sim/vid2sim_gso.zip (gdown 1VOCkvOLDFmJW0Ma6tqwaXW6vSRxhxe49). Outputs: output/vid2sim_eval/<tag>/<obj>.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=${PY:-python}
ALL="backpack bell blocks bus cream elephant grandfather leather lion mario sofa turtle"
ITERS=${ITERS:-100}
COMMON=(--eval-cam 5 --downsample 2 --supersample 2 --contact hertz --friction0 0.5 --substep-scale 3.5 --nonlinear-bonds 1
        --ogden-alpha 2 --ogden-learn 0 --contact-field 1 --cfield-terms 3 --truevol 1 --param Enu --render-mode blend
        --loss feat --pos-scale 2 --phys-weight 0 --prior-weight 0.1 --train-frames 15 --segment 25 --adam-beta2 0.999 --lr-final 1)

setup_obj() {
  o=$1
  [ -d data/vid2sim/ext/$o ] || unzip -q -n data/vid2sim/vid2sim_gso.zip "$o/*" -d data/vid2sim/ext
  [ -f data/vid2sim_powerfoam/${o}_f0/conversion.json ] || $PY experiments/vid2sim_to_powerfoam.py --obj $o
  RUN=output/vid2sim_${o}_spacing8k
  [ -f $RUN/model.pt ] || $PY train.py -c configs/vid2sim.yaml --scene ${o}_f0 --spacing_weight 1e-2 --spacing_kappa .5 \
                            --experiment_name vid2sim_${o}_spacing8k > $RUN.log 2>&1
  DEM=output/vid2sim_dem/$o; mkdir -p output/vid2sim_dem
  [ -f $DEM/geometry.npz ] || $PY experiments/pacnerf_foam_geometry.py --run $RUN --scene-dir data/vid2sim_powerfoam/${o}_f0 \
                                --min-fraction 0.1 --out $DEM > $DEM.geometry.log 2>&1
  $PY experiments/vid2sim_to_springgaus.py --obj $o
}

train_obj() {   # obj E0 nu0 iterations out [extra args]
  o=$1; E0=$2; NU0=$3; IT=$4; OUT=$5; shift 5
  mkdir -p $OUT
  $PY -u experiments/springgaus_train.py material --obj vid2sim_$o --run output/vid2sim_${o}_spacing8k --dem output/vid2sim_dem/$o \
      --E $E0 --nu $NU0 "${COMMON[@]}" --iterations $IT "$@" --out $OUT > $OUT.log 2>&1
}

summary() {   # tag
  $PY - "$1" <<'EOF'
import json, math, sys
from pathlib import Path
import numpy as np
tag = sys.argv[1]
base = Path('output/vid2sim_eval')/tag
rows = []
for d in sorted(p for p in base.iterdir() if (p/'report.json').exists()):
    r = json.loads((d/'report.json').read_text()); cfg = r['config']; b = r['best']
    obj = cfg['obj'].replace('vid2sim_', '')
    gt = json.loads(Path(f'data/springgaus_powerfoam/vid2sim_{obj}_dynamic/cameras.json').read_text())['gt']
    E = float(cfg['E'])*r['E_nu']['E_multiplier']; nu = r['E_nu']['nu']
    rows.append(dict(obj=d.name, E_gt=gt['yms'], E=E, log10_err=abs(math.log10(E/gt['yms'])), nu_gt=gt['prs'], nu=nu,
                     nu_err=abs(nu-gt['prs']), damping=r['multipliers']['damping'], iou_train=b['iou_trainviews_trainframes'],
                     iou_eval=b['iou_evalview_trainframes'], iou_future_train=b['iou_trainviews_futureframes'],
                     iou_future_eval=b['iou_evalview_futureframes']))
print('%-12s %10s %10s %7s %6s %6s %6s %8s %7s %7s' % ('object', 'E_gt', 'E', 'log10e', 'nu_gt', 'nu', 'nu_e', 'damping', 'IoU_tr', 'IoU_fut'))
for x in rows:
    print('%-12s %10.4g %10.4g %7.3f %6.3f %6.3f %6.3f %8.2f %7.3f %7.3f' % (x['obj'], x['E_gt'], x['E'], x['log10_err'], x['nu_gt'], x['nu'],
          x['nu_err'], x['damping'], x['iou_train'], x['iou_future_eval']))
if rows:
    le = np.array([x['log10_err'] for x in rows]); ne = np.array([x['nu_err'] for x in rows])
    print('n=%d  log10 E error: mean %.3f median %.3f | nu error: mean %.3f median %.3f' % (len(rows), le.mean(), np.median(le), ne.mean(), np.median(ne)))
(base/'summary.json').write_text(json.dumps(rows, indent=1))
EOF
}

cmd=${1:-}; shift || true
case $cmd in
  setup) for o in ${@:-$ALL}; do echo "$(date) setup $o"; setup_obj $o; done ;;
  check) for o in ${@:-$ALL}; do
           train_obj $o 1e5 0.3 0 output/vid2sim_eval/check/$o
           $PY -c "
import json, numpy as np
f = np.array(json.load(open('output/vid2sim_eval/check/$o/report.json'))['initial']['iou_per_frame_cam'])
print('$o: frame-0 IoU min over views %.3f, free-fall frames 1-4 mean %.3f' % (f[0].min(), f[1:5].mean()))"
         done ;;
  start_test) for E0 in 2e4 1e5 1e6; do echo "$(date) sofa from E $E0"; train_obj sofa $E0 0.3 $ITERS output/vid2sim_eval/start_test/sofa_E$E0; done
              summary start_test ;;
  all) E0=${1:-1e5}; shift || true
       for o in ${@:-$ALL}; do echo "$(date) train $o from E $E0"; train_obj $o $E0 0.3 $ITERS output/vid2sim_eval/all_E$E0/$o; done
       summary all_E$E0 ;;
  summary) summary $1 ;;
  *) sed -n 2,12p "$0"; exit 1 ;;
esac
