#!/bin/bash
# stage0_run.sh -- run the Stage 0 validation package.
#
# Lives in dft_gsfe_scan/ (tracked) but must execute in gsfe_scan_inputs/
# (untracked run area), because every generated input refers to pseudo_dir and
# outdir by relative path. The script therefore locates the run directory
# itself rather than relying on where it was invoked from -- calling it from
# the wrong place would otherwise fail late, after QE has already started, with
# a message about missing pseudopotentials.
#
#   bash dft_gsfe_scan/stage0_run.sh          # from atomistic_bridge_placeholder/
#   bash ../dft_gsfe_scan/stage0_run.sh       # from gsfe_scan_inputs/
#
# Order matters: the nine physical runs come first, so that if the machine is
# needed for something else the verdict on SOC and the dipole correction is
# already available and only the convergence probes are missing. Completed runs
# are skipped, so the script is safe to interrupt and restart.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "$SCRIPT_DIR/../gsfe_scan_inputs" 2>/dev/null && pwd)" || {
  echo "Cannot find gsfe_scan_inputs/ next to $SCRIPT_DIR" >&2
  exit 1
}
cd "$RUN_DIR" || exit 1

if [ ! -d pseudo_rel ]; then
  echo "pseudo_rel/ missing in $RUN_DIR -- see PSEUDO_README.md" >&2
  exit 1
fi

NP=${NP:-8}
export OMP_NUM_THREADS=1

D=stage0
if [ ! -d "$D" ]; then
  echo "$D/ missing -- run: python3 ../dft_gsfe_scan/stage0_generate.py --outdir $D" >&2
  exit 1
fi
mkdir -p "$D/pdos" "$D/pot" pdos pot tmp_stage0

echo "run directory : $RUN_DIR"
echo "MPI ranks     : $NP"
echo

run_scf () {
  local name=$1
  if [ -f "$D/$name.out" ] && grep -aq "JOB DONE" "$D/$name.out"; then
    echo "  $name: already done, skipping"
    return
  fi
  echo "  $name: $(date +%H:%M:%S) ..."
  mpirun -np "$NP" --oversubscribe pw.x -in "$D/$name.in" > "$D/$name.out" 2>&1
  if grep -aq "JOB DONE" "$D/$name.out"; then
    local t conv
    t=$(grep -aoP 'PWSCF\s+:\s+\K.*WALL' "$D/$name.out" | tail -1)
    conv=$(grep -ac "convergence has been achieved" "$D/$name.out")
    # JOB DONE alone is not success: QE also prints it after "convergence NOT
    # achieved", and those eigenvalues are meaningless.
    if [ "$conv" -eq 0 ]; then
      echo "      *** completed but SCF did NOT converge -- $t"
    else
      echo "      done $(date +%H:%M:%S)   $t"
    fi
  else
    echo "      *** FAILED: $(grep -a -A1 'Error in routine' "$D/$name.out" | tail -1)"
  fi
}

echo "=== physical variants (B, C, D on AA, AB, BA) ==="
for s in AA AB BA; do
  for v in B C D; do
    run_scf "${s}_${v}"
  done
done

echo
echo "=== layer projections ==="
for s in AA AB BA; do
  for v in B C D; do
    n="${s}_${v}"
    [ -f "$D/$n.out" ] || continue
    grep -aq "JOB DONE" "$D/$n.out" || continue
    if [ -f "$D/$n.projwfc.out" ] && grep -aq "JOB DONE" "$D/$n.projwfc.out"; then
      echo "  $n: already projected, skipping"
      continue
    fi
    mpirun -np 4 projwfc.x -in "$D/$n.projwfc.in" > "$D/$n.projwfc.out" 2>&1
    echo "  $n: spilling $(grep -aoP 'Spilling Parameter:\s+\K[\d.]+' "$D/$n.projwfc.out" | tail -1)"
  done
done

echo
echo "=== electrostatic profile (C vs D, to locate the vacuum level) ==="
for n in AB_C AB_D; do
  [ -f "$D/$n.pp.in" ] || continue
  if ! grep -aq "JOB DONE" "$D/$n.out" 2>/dev/null; then
    echo "  $n: SCF missing or failed, skipping"
    continue
  fi
  mpirun -np 4 pp.x -in "$D/$n.pp.in" > "$D/$n.pp.out" 2>&1
  f="pot/${n}_vz.dat"
  if [ -f "$f" ]; then
    echo "  $n: $(wc -l < "$f") points written"
  else
    echo "  $n: no output -- $(grep -a -A1 'Error in routine' "$D/$n.pp.out" | tail -1)"
  fi
done

echo
echo "=== convergence probes ==="
for n in conv_ecut85 conv_k12 conv_k15 conv_vac30 conv_vac35; do
  run_scf "$n"
done

echo
echo "ALL DONE $(date +%H:%M:%S)"
echo "Analyse with: python3 ../dft_gsfe_scan/stage0_collect.py $D/"
