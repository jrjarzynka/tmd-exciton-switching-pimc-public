# Fourier / normal-mode normalization note — 2026-09-08

The development API distinguishes raw one-sided rFFT coefficients from physical
orthonormal real normal-mode coordinates.

For `q = rfft(path, norm="ortho")` and real `path`:

```text
sum_j |r_j|^2
= |q_0|^2
+ [|q_P/2|^2 for even P]
+ 2 sum_interior |q_k|^2.
```

The same one-sided multiplicities enter the nearest-neighbour spring sum:

```text
sum_j |r_(j+1)-r_j|^2
= sum_k w_k * 4 sin^2(pi k/P) * |q_k|^2,
```

with `w_0=1`, `w_Nyquist=1` for even `P`, and `w_k=2` otherwise.

Therefore:

- raw stored-half-spectrum `|q_k|^2` is not reported as a physical orthonormal
  mode power for interior bins;
- `centroid_free_mode_powers()` now returns Parseval-weighted contributions;
- proposal amplitudes intended to have direct physical/analytic meaning should
  use the real orthonormal cosine/sine coordinates:
  `c_k=sqrt(2) Re(q_k)`, `s_k=-sqrt(2) Im(q_k)` for interior modes;
- centroid and even-P Nyquist modes have no sine coordinate;
- odd `P` has no Nyquist self-conjugate bin.

This clarification does not alter staging/Brownian-bridge physics or any target
action. It prevents ambiguous interpretation of Fourier powers, proposal scales,
analytic mode variances, and ESS/IAT comparisons.
