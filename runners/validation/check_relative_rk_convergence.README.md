# RK convergence checker

Install `check_relative_rk_convergence.py` in the project `scripts/` directory.

```bash
cd "/home/jaro/Projects/v1.6 2D relative-coordinate exciton PIMC validation"
cp /path/to/check_relative_rk_convergence.py scripts/
```

Run across multiple RK validation directories:

```bash
python3 scripts/check_relative_rk_convergence.py \
  --results_glob 'results/relative_rk_*' \
  --outdir results/relative_rk_convergence_check
```

Or name directories explicitly:

```bash
python3 scripts/check_relative_rk_convergence.py \
  --results_dirs \
    results/relative_rk_T20_stage1 \
    results/relative_rk_T20_P256 \
  --outdir results/relative_rk_convergence_check
```

The checker keeps each directory as a separate **tag**. Monotonicity is tested
only within each `(tag, temperature)` group, exactly as requested. A tag can
contain any subset of temperatures and P values.

## Outputs

- `combined_relative_rk_convergence.csv`: all source rows with derived metrics.
- `rk_point_checks.csv`: SEM/bias, wall and tau-A status for every `(tag,T,P)`.
- `rk_validation_by_tag_temperature.csv`: the six criteria for every `(tag,T)`.
- `rk_validation_by_tag.csv`: one overall status per result directory.
- `rk_convergence_validation_report.txt`: readable detailed report.
- `rk_convergence_validation_summary.json`: machine-readable summary.

## Status logic

- `PASS`: every hard gate passes and the convergence evidence is strong.
- `WARN`: no invalid physics was found, but monotonicity could not be tested
  (only one P) or SEM is larger than the remaining bias (`WEAK`).
- `FAIL`: largest-P r2 misses the configured threshold, an error reverses with
  P, wall effects are too large, table diagnostics fail/missing, or tau*A >= 2.

Default thresholds can all be changed from the CLI. Useful options include:

```text
--r2_threshold_percent 2
--table_max_relative_error 1e-3
--sem_bias_max_ratio 1
--sem_bias_strong_ratio 0.5
--wall_max_energy_meV 0.01
--wall_max_probability 1e-6
--tau_strong_max 1
--tau_invalid_min 2
--monotonic_rtol 0
--allow_missing_table_diagnostics
--fail_on_warn
```

Exit status is `0` for PASS/WARN, `2` for a hard FAIL, and `3` for invalid
input. Add `--fail_on_warn` for CI-style strictness.
