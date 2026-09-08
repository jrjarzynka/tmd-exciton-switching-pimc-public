# Ring-polymer move architecture: COM now, explicit electron-hole next

The staging/Fourier upgrade must not be defined by the Paper-1 COM model.  The
COM sampler is the first consumer of reusable ring-polymer move primitives.

## 1. Reusable layer

The reusable layer contains no moire, Coulomb or particle identity:

- periodic imaginary-time segment indexing;
- exact free Brownian-bridge proposal;
- bridge validation/log density;
- one-sided rFFT transform/inverse with explicit Parseval multiplicities;
- real orthonormal cosine/sine normal-mode coordinates;
- literal low-mode shifts for deterministic tests.

For particle `a`, the staging proposal receives only

```
path_a[P,D]
lambda_a = hbar^2/(2 m_a)
tau
segment
```

and uses one-link variance

```
sigma_link,a^2 = 2 lambda_a tau.
```

## 2. Paper-1 single-polymer COM consumer

The existing periodic COM kernel evaluates

```
S = S_spring + tau sum_j V_periodic(r_j).
```

An exact free bridge absorbs the segment spring probability, leaving

```
Delta S = tau sum_interior [V(new_j)-V(old_j)].
```

Coordinates stay unwrapped on the infinite Cartesian cover.  Only potential
lookup is reduced modulo the primitive lattice.

## 3. Explicit two-polymer electron-hole consumer

The target architecture is

```
S = S_spring,e + S_spring,h
  + tau sum_j [V_e(r_e,j) + V_h(r_h,j) + V_eh(|r_e,j-r_h,j|)].
```

with distinct masses and therefore distinct `lambda_e`, `lambda_h`.

### Electron segment update

Hold the hole path fixed and generate an electron Brownian bridge using
`2*lambda_e*tau`.  The spring contribution is absorbed by the proposal and

```
Delta S_e = tau sum_interior [
    V_e(r'_e,j)-V_e(r_e,j)
  + V_eh(|r'_e,j-r_h,j|)-V_eh(|r_e,j-r_h,j|)
].
```

### Hole segment update

Likewise, using `2*lambda_h*tau`,

```
Delta S_h = tau sum_interior [
    V_h(r'_h,j)-V_h(r_h,j)
  + V_eh(|r_e,j-r'_h,j|)-V_eh(|r_e,j-r_h,j|)
].
```

This works for soft-Coulomb, screened RK and interlayer radial interactions as
long as the interaction evaluator is the one validated for the Hamiltonian.

## 4. Fourier / normal-mode path

A direct mode update can act on electron or hole normal modes independently.
The stored `rfft(..., norm="ortho")` half-spectrum is convenient but must not
be treated as an ordinary Euclidean-orthonormal vector: interior bins represent
conjugate `+k/-k` pairs and carry Parseval weight 2.  For reported mode powers
and spring-action reconstruction we therefore use explicit one-sided weights.
For proposal amplitudes and analytic variances, the preferred API is the real
orthonormal cosine/sine basis (`c_k=sqrt(2) Re q_k`, `s_k=-sqrt(2) Im q_k` for
interior modes; centroid and even-P Nyquist are self-conjugate).  This transform
has a constant Jacobian, so a symmetric shift in those real coordinates is
accepted with the full target-action change.

For the two-body problem we should later test two additional proposal families:

1. independent electron/hole low-mode moves;
2. correlated common/relative moves, e.g. mass-aware combinations that mainly
   change the pair COM or the relative e-h coordinate.

The relative-mode family is especially attractive because the interaction is a
function of `r_e-r_h`.

## 5. Implementation rule

Do not force one monolithic generic Numba kernel.  Keep the *proposal
mathematics* reusable and allow optimized consumer kernels to evaluate their
own potential deltas:

```
staging.py / normal_modes.py       reusable mathematics
single-body periodic kernel        Paper-1 COM consumer
two-body kernel                    electron-hole consumer
```

This avoids Python callbacks inside Numba while preventing the bridge/Fourier
mathematics from being reimplemented inconsistently.
