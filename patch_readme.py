#!/usr/bin/env python3
"""
Add the seed-to-seed uncertainty campaign to the README reproducibility table.

The Data availability statement promises a mapping from every validation
section to the script that reproduces it. After the v4 campaign, Table 4,
Fig. 4 and the straddling fraction come from new scripts that the table does
not list, so that promise is currently not kept.

Three rows are added and two existing rows are qualified, so that a reader can
tell which numbers are single-chain and which are eight-chain means.

Run from the repository root. Backs the file up first.
"""
import datetime
import shutil
import sys

PATH = "README.md"

NEW_ROWS = (
    "| Validation I: harmonic benchmark, seed-to-seed (Table 4, Fig. 2) | "
    "`validation_tests/run_seed_uncertainty_campaign.py` (part B) | "
    "Eight independent chains per temperature; writes `harmonic_seed_check.csv`. "
    "Supersedes the single-chain values for Table 4 |\n"
    "| Validation II: occupation scan, seed-to-seed (Fig. 4) | "
    "`validation_tests/test3_double_well_v4_multiseed.py` | "
    "Eight independent chains per field point; writes `occ_scan_v4_multiseed.csv` "
    "and `occ_scan_v4_summary.csv`. Supersedes `test3_double_well_v3.py` for the "
    "occupations and their error bars |\n"
    "| Validation II: straddling fraction, seed-to-seed (Sec. 5.2) | "
    "`validation_tests/run_seed_uncertainty_campaign.py` (part A) | "
    "Eight independent chains; writes `straddling_multiseed.csv`. Supersedes the "
    "single-chain 25.6% previously quoted |\n"
    "| Figures 2 and 4 | `validation_tests/plot_figs_v4.py` | "
    "Regenerates both figures with seed-to-seed error bars from the campaign CSVs |\n"
    "| Graphical abstract | `validation_tests/make_graphical_abstract_v4.py` | "
    "Built from the campaign CSVs and `samples_symmetric_Ex0.npy`; no hand-entered "
    "numbers |\n"
)

EDITS = [
    # Insert the new rows directly after the existing Validation II row, so the
    # superseding scripts sit next to what they supersede.
    ("insert v4 rows after the Validation II row",
     "| Validation II: field-driven double-well relocation | "
     "`validation_tests/test3_double_well_v3.py` | Symmetric double well, "
     "occupation crossover at `Ex=0` |\n",
     "| Validation II: field-driven double-well relocation (single chain) | "
     "`validation_tests/test3_double_well_v3.py` | Symmetric double well, "
     "occupation crossover at `Ex=0`. One chain per field point; retained for "
     "provenance, superseded by the v4 campaign below |\n" + NEW_ROWS),

    # Table 5 still comes from test2, but Table 4 no longer does; say which.
    ("qualify the single-chain harmonic row",
     "| Validation I: harmonic benchmark | "
     "`validation_tests/test1_single_well.py` | PIMC vs. exact",
     "| Validation I: harmonic benchmark (single chain) | "
     "`validation_tests/test1_single_well.py` | Single-chain reference run; "
     "PIMC vs. exact"),

    # Both IAT scripts are listed as though either backs Table 3. Only v2 does.
    ("distinguish the two IAT scripts",
     "| Sec. 3.2: autocorrelation / staging | "
     "`validation_tests/test5_IAT_scaling.py`, `test5_IAT_scaling_v2.py` | "
     "`z_local` vs. `z_staging` power-law fit |",
     "| Sec. 3.2: autocorrelation / staging (Table 3, Fig. 1) | "
     "`validation_tests/test5_IAT_scaling_v2.py` | `z_local` vs. `z_staging` "
     "power-law fit over four seeds. `test5_IAT_scaling.py` is the superseded "
     "single-seed version, retained for provenance; its stored exponents "
     "(`z_staging = 0.0905`) are **not** the published values |"),
]


def main():
    try:
        text = open(PATH, encoding="utf-8").read()
    except FileNotFoundError:
        sys.exit(f"{PATH} not found -- run from the repository root")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.copy(PATH, f"{PATH}.bak-{stamp}")
    print(f"backup: {PATH}.bak-{stamp}\n")

    ok = 0
    for label, old, new in EDITS:
        n = text.count(old)
        if n != 1:
            print(f"  NOT APPLIED: {label}  (matches={n})")
            continue
        text = text.replace(old, new, 1)
        ok += 1
        print(f"  applied: {label}")

    open(PATH, "w", encoding="utf-8").write(text)
    print(f"\n{ok}/{len(EDITS)} applied")


if __name__ == "__main__":
    main()
