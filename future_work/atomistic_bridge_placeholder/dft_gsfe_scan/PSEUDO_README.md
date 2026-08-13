# Pseudopotentials for Stage 0

Not tracked. Download into `gsfe_scan_inputs/pseudo_rel/` from
https://pseudopotentials.quantum-espresso.org/upf_files/ :

    Mo.rel-pbe-spn-kjpaw_psl.1.0.0.UPF    Mo.pbe-spn-kjpaw_psl.1.0.0.UPF
    Se.rel-pbe-dn-kjpaw_psl.1.0.0.UPF     Se.pbe-dn-kjpaw_psl.1.0.0.UPF
    W.rel-pbe-spn-kjpaw_psl.1.0.0.UPF     W.pbe-spn-kjpaw_psl.1.0.0.UPF

Each rel-/scalar pair shares z_valence (14/16/14), generator and suggested
cutoffs, differing only in whether the j-resolved channels are present. That
matters: QE refuses a fully relativistic pseudopotential unless lspinorb is on
("Fully relativistic PPs, need spin-orbit calc."), so a noncollinear-without-SOC
control is impossible and variant B must use the scalar files instead.

The production 49-point scan used a different set entirely (Mo_ONCV_PBE,
Se_pbe_v1.uspp, W_pbe_v1.2.uspp; Se with 6 valence electrons rather than 16).
The two disagree on the interlayer distance by 0.66 A (5.92 vs 6.58 A) and on
the registry corrugation by roughly a factor of two (538 vs 274 meV). Neither is
wrong: PBE alone does not bind the layers at all, so the minimum comes entirely
from the balance between PBE repulsion and the D3 attraction, and its position
is correspondingly sensitive to the valence density -- that is, to the
pseudopotential.
