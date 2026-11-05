#!/bin/bash
# The lion's material identification on a server, unattended (RUN_LION_INVERSE.md explains what runs).
#
#   nohup bash run_lion_inverse.sh > lion_inverse.log 2>&1 &        everything, in order (or inside tmux)
#   bash run_lion_inverse.sh env|check|synth|real|eval|summary      one stage
#
# Stages: env (packages, GPU) -> check (frame 0 against the masks, a forward to frame 8 against this machine's numbers;
# stops on a mismatch) -> synth (LM on observations from our own model at a known material: can it find it?) -> real
# (LM on the Vid2Sim video) -> eval (frames 0-23 at the fitted material and at the ground-truth one: IoU, errors) ->
# summary (summary.txt, and lion_inverse_results.tgz with everything to copy back). A finished stage is skipped on a
# rerun; an interrupted LM resumes from its last accepted point. Settings by environment: PY (python), JOBS (runs at
# once; default from the GPU's free memory, about 4.5 GB each, at most 7), FOAMSET (lion, default: the old foam;
# lion_vfc: the 11-06 foam whose tail is a chain of single spheres, with CHAIN=1 rigid chain cells by default),
# OUT (default out_inverse, out_inverse_vfc for lion_vfc).
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-python}
FOAMSET=${FOAMSET:-lion}                                       # lion: fourier_lion_8k_round_clean; lion_vfc: the 11-06 round_vertex_faces_chain foam
if [ "$FOAMSET" = lion_vfc ]; then CHAIN=${CHAIN:-1}; OUT=${OUT:-out_inverse_vfc}; else CHAIN=${CHAIN:-0}; OUT=${OUT:-out_inverse}; fi
FA="--foam inverse_data/$FOAMSET/foam --chain $CHAIN"         # (every lion_inverse.py call: the foam, the rigid chain cells)
mkdir -p "$OUT"
LI="$PY -u free_surface/lion_inverse.py"
# the materials (ln E, nu, ln beta): Vid2Sim's ground truth E 10648.6, nu 0.3913; synthetic truth beta 0.08
GT_LNE=9.273184; GT_NU=0.3912951704721983
SYN_TRUTH="9.273184 0.3912951704721983 -2.525729"          # E 10648.6, nu 0.391, beta 0.08
SYN_START="10.371796 0.30 -1.897120"                         # E x3, nu 0.30, beta 0.15
REAL_START="10.371796 0.30 -2.302585"                        # E x3, nu 0.30, beta 0.10
say() { echo "== $(date '+%m-%d %H:%M:%S') $*"; }
die() { echo "!! $(date '+%m-%d %H:%M:%S') $*"; exit 1; }

jobs_default() {
  local free
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -z "$free" ]; then echo 2; return; fi
  local j=$(( free / 4500 )); [ $j -gt 7 ] && j=7; [ $j -lt 1 ] && j=1; echo $j
}
JOBS=${JOBS:-$(jobs_default)}

stage_env() {
  say "env: python $($PY -c 'import sys; print(sys.version.split()[0])'), JOBS $JOBS"
  if ! $PY -c "import torch, warp, numpy, scipy, numba, PIL, yaml, tqdm" 2>/dev/null; then
    say "env: a package is missing, installing requirements-inverse.txt (torch is not touched)"
    $PY -m pip install -r requirements-inverse.txt || die "pip install failed"
  fi
  $PY -c "
import torch, warp, numpy, scipy, numba
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')
print('warp', warp.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'numba', numba.__version__)
assert torch.cuda.is_available(), 'no CUDA device'" || die "env: no usable GPU / packages"
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
}

stage_check() {
  if [ -f "$OUT/check.ok" ]; then say "check: done before, skipped"; return; fi
  say "check 1/2: frame 0, our silhouette against the masks"
  $LI check $FA > "$OUT/check.log" 2>&1 || die "check failed, see $OUT/check.log"
  grep -E "^V[0-9]+:|mean" "$OUT/check.log"
  $PY - "$OUT/check.log" $FOAMSET <<'EOF' || die "check 1/2: IoU differs from this machine's (see above)"
import re, sys
ref = dict(lion=[0.9607, 0.9178, 0.9608, 0.9640, 0.9629, 0.9772, 0.9637, 0.9380, 0.9604, 0.9574, 0.9580, 0.9243],     # 11-06, RTX 5070 Ti
           lion_vfc=[0.9929, 0.9953, 0.9942, 0.9942, 0.9920, 0.9942, 0.9945, 0.9916, 0.9939, 0.9941, 0.9932, 0.9923])[sys.argv[2]]
got = [float(m.group(1)) for m in re.finditer(r'^V\d+: IoU ([0-9.]+)', open(sys.argv[1]).read(), re.M)]
bad = [(k, g, r) for k, (g, r) in enumerate(zip(got, ref)) if abs(g-r) > 2e-3]
print('IoU against this machine: %d views, largest difference %.4f' % (len(got), max(abs(g-r) for g, r in zip(got, ref))))
sys.exit(1 if len(got) != 12 or bad else 0)
EOF
  say "check 2/2: a forward at the ground truth to frame 8 (about a minute after the first compile)"
  $LI run $FA --theta $GT_LNE $GT_NU -2.995732 --frames 8 --out "$OUT/smoke.npz" > "$OUT/smoke.log" 2>&1 || die "the smoke forward failed, see $OUT/smoke.log"
  grep -E "frame [678] |done" "$OUT/smoke.log"
  $PY - "$OUT/smoke.log" $FOAMSET <<'EOF' || die "check 2/2: the trajectory differs from this machine's (see above)"
import re, sys
s = open(sys.argv[1]).read()
z = {int(k): float(v) for k, v in re.findall(r'frame (\d+) \(t [0-9.]+ s, [0-9.]+ s\): com z (-?[0-9.]+) m', s)}
ref = dict(lion={5: 0.1590, 6: 0.0158, 7: -0.1179, 8: -0.1224},                       # 11-06, RTX 5070 Ti
           lion_vfc={5: 0.1645, 6: 0.0191, 7: -0.1254, 8: -0.1343})[sys.argv[2]]       # (with --chain 1)
d = max(abs(z.get(k, 9.)-v) for k, v in ref.items())
print('com z at frames 5-8 against this machine: largest difference %.4f m' % d)
sys.exit(0 if d <= 1e-3 else 1)
EOF
  touch "$OUT/check.ok"; say "check: passed"
}

run_lm() {   # name obs start [extra]
  local name=$1 obs=$2 start=$3; shift 3
  if [ -f "$OUT/$name/final_theta.json" ]; then say "$name: done before, skipped"; return; fi
  local res=""; [ -f "$OUT/$name/lm.json" ] && res="--resume"
  say "$name: LM ($obs observations), $JOBS runs at once ${res:+(resuming)}"
  # (the LM's own log in $OUT/$name.log; its progress lines also here, as they come)
  $LI lm $FA --obs $obs --start $start --frames 15 --jobs $JOBS --out "$OUT/$name" $res "$@" 2>&1 | tee -a "$OUT/$name.log" \
    | grep --line-buffered -E "^it |runs \(|forward's error|identifiability|stop|final|Traceback|Error"
  local rc=${PIPESTATUS[0]}
  [ $rc -eq 0 ] && [ -f "$OUT/$name/final_theta.json" ] || die "$name: LM stopped with code $rc, see $OUT/$name.log (rerun this script to resume)"
  say "$name: finished"
}

stage_synth() {
  run_lm synth synth "$SYN_START" --truth $SYN_TRUTH
  $PY - "$OUT/synth/final_theta.json" <<'EOF' | tee "$OUT/synth_verdict.txt"
import json, math, sys
t = json.load(open(sys.argv[1]))
E, nu, b = t['E'], t['nu'], t['beta']
eE, enu, eb = abs(math.log10(E/10648.614517785627)), abs(nu-0.3912951704721983), abs(b/0.08-1)
ok = eE < .02 and enu < .01 and eb < .1
print('synthetic: E %.1f (truth 10648.6, log10 error %.4f), nu %.4f (0.3913, error %.4f), beta %.4f (0.08, %.1f %%) -> %s' % (
    E, eE, nu, enu, b, 100*eb, 'PASS' if ok else 'FAIL (the real stage still runs; judge its result with this in mind)'))
EOF
}

stage_real() { run_lm real real "$REAL_START"; }

stage_lmsmoke() {   # (not part of `all`) one LM iteration on 7 frames of synthetic data: does the first step stay inside the cap?
  rm -rf "$OUT/lmsmoke" "$OUT/lmsmoke.log"
  say "lmsmoke: one LM iteration, synthetic, frames 0-7 (about 20-30 min)"
  $LI lm $FA --obs synth --truth $SYN_TRUTH --start $SYN_START --frames 7 --iters 1 --jobs $JOBS --out "$OUT/lmsmoke" 2>&1 | tee "$OUT/lmsmoke.log" \
    | grep --line-buffered -E "^it |runs \(|forward's error|identifiability|stop|final|Traceback|Error"
  $PY - "$OUT/lmsmoke/lm.json" <<'EOF' || die "lmsmoke: no LM history"
import json, sys
h = json.load(open(sys.argv[1]))
a, b = h[0], h[1]
d = [abs(x-y) for x, y in zip(b['phi'], a['phi'])]
print('lmsmoke: first step %s; change of (ln mu, ln K/mu, ln beta) %s, largest %.3f (cap 0.3); nu %.4f -> %.4f (truth 0.3913); E %.1f -> %.1f (truth 10648.6)' % (
    'accepted' if b['accepted'] else 'rejected', ['%.3f' % v for v in d], max(d), a['nu'], b['nu'], a['E'], b['E']))
sys.exit(0 if max(d) <= .3+1e-9 else 1)
EOF
}

stage_eval() {
  if [ ! -f "$OUT/real_eval/eval.json" ]; then
    [ -f "$OUT/real/final_theta.json" ] || die "eval: no $OUT/real/final_theta.json (run the real stage first)"
    local th; th=$($PY -c "import json; print(' '.join('%.17g' % v for v in json.load(open('$OUT/real/final_theta.json'))['theta']))")
    say "eval: frames 0-23 at the fitted material ($th)"
    $LI eval $FA --theta $th --out "$OUT/real_eval" > "$OUT/real_eval.log" 2>&1 || die "eval failed, see $OUT/real_eval.log"
    grep "^eval" "$OUT/real_eval.log"
  fi
  if [ ! -f "$OUT/gt_eval/eval.json" ]; then
    say "eval: frames 0-23 at the ground-truth E, nu (beta 0.05), for reference"
    $LI eval $FA --theta $GT_LNE $GT_NU -2.995732 --out "$OUT/gt_eval" > "$OUT/gt_eval.log" 2>&1 || say "eval at the ground truth failed (see $OUT/gt_eval.log), going on"
    grep "^eval" "$OUT/gt_eval.log" || true
  fi
}

stage_summary() {
  {
    echo "lion inverse ($FOAMSET, chain $CHAIN), $(date '+%Y-%m-%d %H:%M'), branch $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD), $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    cat "$OUT/synth_verdict.txt" 2>/dev/null
    for n in synth real; do [ -f "$OUT/$n/lm.json" ] && $PY -c "
import json; h=[e for e in json.load(open('$OUT/$n/lm.json')) if e['accepted']]
print('$n: %d accepted iterations; last: E %.1f, nu %.4f, beta %.4f, RMS %.3f px, log10 E error %.3f, nu error %.4f' % (len(h), h[-1]['E'], h[-1]['nu'], h[-1]['beta'], h[-1]['rms'], h[-1]['log10E_err'], h[-1]['nu_err']))
import os
if os.path.exists('$OUT/$n/identifiability.json'):
    q=json.load(open('$OUT/$n/identifiability.json'))[-1]['residual']; print('   identifiability there: sigma(log10 E) %.4f, sigma(nu) %.4f, corr(E, nu) %+.3f' % (q['sigma_log10E'], q['sigma_nu'], q['corr_E_nu']))"; done
    grep -h "^eval" "$OUT/real_eval.log" "$OUT/gt_eval.log" 2>/dev/null | sed -e '1s/^/fitted:  /' -e '2s/^/GT E,nu: /'
  } | tee "$OUT/summary.txt"
  tar czf lion_inverse_results.tgz "$OUT"/*.log "$OUT"/*.txt "$OUT"/*/lm.json "$OUT"/*/identifiability.json "$OUT"/*/final_theta.json "$OUT"/*/eval.json "$OUT"/*/eval_frames.npz 2>/dev/null
  say "summary: $OUT/summary.txt; copy back lion_inverse_results.tgz ($(du -h lion_inverse_results.tgz | cut -f1))"
}

case "${1:-all}" in
  env) stage_env ;;
  check) stage_check ;;
  synth) stage_synth ;;
  real) stage_real ;;
  lmsmoke) stage_lmsmoke ;;
  eval) stage_eval ;;
  summary) stage_summary ;;
  all) stage_env; stage_check; stage_synth; stage_real; stage_eval; stage_summary; say "all done" ;;
  *) echo "usage: bash run_lion_inverse.sh [all|env|check|lmsmoke|synth|real|eval|summary]"; exit 2 ;;
esac
