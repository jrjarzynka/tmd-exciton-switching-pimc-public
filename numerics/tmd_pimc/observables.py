import numpy as np


def centroids(samples: np.ndarray) -> np.ndarray:
    """Centroid of each ring polymer: shape (N_samples, 2)."""
    return samples.mean(axis=1)


def all_beads(samples: np.ndarray) -> np.ndarray:
    """All bead positions flattened: shape (N_samples * P, 2)."""
    return samples.reshape(-1, 2)


def radius2(points: np.ndarray) -> np.ndarray:
    """Squared distance from origin for each row: shape (N,)."""
    return np.einsum("ij,ij->i", points, points)


# ---------------------------------------------------------------------------
# Quantum spatial estimators  (use all beads — correct quantum average)
# ---------------------------------------------------------------------------

def r2_mean_pimc(samples: np.ndarray) -> float:
    """
    Mean <r²> over all beads — correct quantum spatial estimator.

    Measures mean squared distance from the coordinate origin.
    Note: large values arise when the exciton sits far from (0,0);
    use r2_spread_pimc for the intrinsic quantum delocalisation width.
    """
    return float(np.mean(radius2(all_beads(samples))))


def r2_spread_pimc(samples: np.ndarray) -> float:
    """
    Mean squared spread of beads around their own centroid.

    This is the intrinsic quantum delocalisation width, independent of
    where in space the exciton is localised.  Use this to measure how
    'quantum-smeared' the exciton is around its equilibrium position.

        spread² = <|r_bead - r_centroid|²>
    """
    cents = centroids(samples)                        # (N, 2)
    delta = samples - cents[:, np.newaxis, :]         # (N, P, 2)
    return float(np.mean(np.einsum("ijk,ijk->ij", delta, delta)))


def r2_mean_centroid(samples: np.ndarray) -> float:
    """Mean <r²> of centroids — localisation of ring-polymer centre of mass."""
    return float(np.mean(radius2(centroids(samples))))


# ---------------------------------------------------------------------------
# RMS convenience wrappers
# ---------------------------------------------------------------------------

def r_rms_bead(samples: np.ndarray) -> float:
    """RMS distance of all beads from origin."""
    return float(np.sqrt(r2_mean_pimc(samples)))


def r_rms_centroid(samples: np.ndarray) -> float:
    """RMS distance of centroids from origin."""
    return float(np.sqrt(r2_mean_centroid(samples)))


def r_rms_spread(samples: np.ndarray) -> float:
    """RMS quantum spread of beads around their centroid."""
    return float(np.sqrt(r2_spread_pimc(samples)))


# ---------------------------------------------------------------------------
# Switching / localisation observables
# ---------------------------------------------------------------------------

def mean_position(samples: np.ndarray) -> np.ndarray:
    """
    Mean centroid position vector [x, y] in nm.

    Primary observable for detecting lateral excitonic switching:
    a sign change in mean_position[0] (mean_x) as Ex reverses indicates
    the exciton has switched from one moiré minimum to the other.
    """
    return centroids(samples).mean(axis=0)


def switching_contrast(samples_neg: np.ndarray,
                       samples_pos: np.ndarray) -> float:
    """
    Normalised switching contrast along x between two field polarities.

        C = (mean_x_pos - mean_x_neg) / (|mean_x_pos| + |mean_x_neg|)

    C ≈ +1 : clean switching (exciton moves right for positive field)
    C ≈  0 : no switching
    C ≈ -1 : reversed switching
    """
    x_neg = mean_position(samples_neg)[0]
    x_pos = mean_position(samples_pos)[0]
    denom = abs(x_pos) + abs(x_neg)
    if denom == 0.0:
        return 0.0
    return float((x_pos - x_neg) / denom)



# ---------------------------------------------------------------------------
# Compact per-sample time series for autocorrelation diagnostics
# ---------------------------------------------------------------------------

def r2_time_series(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return total, centroid, and internal-spread r² for every saved sample.

    The returned arrays all have shape ``(N_samples,)`` and satisfy, up to
    floating-point roundoff,

        r2_total_series = r2_centroid_series + r2_spread_series.

    They are much smaller than storing the full ``(N_samples, P, 2)`` path and
    are therefore the preferred output for integrated-autocorrelation analysis.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 3 or samples.shape[2] != 2:
        raise ValueError("samples must have shape (N_samples, P, 2)")

    cents = samples.mean(axis=1)
    total = np.mean(np.einsum("ijk,ijk->ij", samples, samples), axis=1)
    centroid = np.einsum("ij,ij->i", cents, cents)
    spread = total - centroid
    return total, centroid, spread
