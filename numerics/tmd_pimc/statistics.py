import numpy as np
from typing import Optional

def autocorrelation(x: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """
    Zwraca znormalizowaną funkcję autokorelacji (ACF) dla 1D serii czasowej.
    Wykorzystuje FFT dla złożoności O(N log N).
    Zwraca rho[0..L], gdzie L = min(n-1, max_lag) lub n-1 gdy max_lag is None.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array")
    n = x.size
    if n == 0:
        return np.array([], dtype=np.float64)

    # centrowanie
    x_centered = x - x.mean()

    # wariancja (sum of squares)
    ss = np.dot(x_centered, x_centered)  # r[0] przed normalizacją
    if np.isclose(ss, 0.0):
        # wariancja praktycznie zerowa: rho(0)=1, pozostałe 0
        L = n - 1 if max_lag is None else min(n - 1, int(max_lag))
        out = np.zeros(L + 1, dtype=np.float64)
        out[0] = 1.0
        return out

    # rozmiar FFT: najmniejsza potęga 2 >= 2*n-1
    n_fft = 1 << int(np.ceil(np.log2(2 * n - 1)))

    # FFT i korelacja (autokowariancja)
    fx = np.fft.fft(x_centered, n=n_fft)
    acov = np.fft.ifft(fx * np.conjugate(fx)).real[:n]

    # normalizacja przez r[0] = ss
    rho = acov / ss
    rho[0] = 1.0  # zapewnienie dokładnego 1.0

    # przycinanie do max_lag
    if max_lag is not None:
        if max_lag < 0:
            raise ValueError("max_lag must be non-negative or None")
        L = min(n - 1, int(max_lag))
        rho = rho[: L + 1]

    return rho

def integrated_autocorrelation_time(rho: np.ndarray, cutoff: str = "first_negative") -> float:
    """
    Oblicza zintegrowany czas autokorelacji:
    tau_int = 0.5 + sum_{k=1}^{K} rho(k)
    Metoda cutoff 'first_negative' sumuje do pierwszego k>0, gdzie rho(k) < 0 (nie włączając tego k).
    """
    rho = np.asarray(rho, dtype=np.float64)
    if rho.size == 0:
        return 0.5
    if rho.size == 1:
        return 0.5

    if cutoff == "first_negative":
        # znajdź pierwszy indeks >0 gdzie rho < 0
        neg = np.where(rho[1:] < 0)[0]
        if neg.size > 0:
            k_max = neg[0] + 1  # +1 bo przeszliśmy rho[1:]
        else:
            k_max = rho.size
        s = np.sum(rho[1:k_max])
        tau = 0.5 + float(s)
        # zabezpieczenie przed ujemnym tau (teoretycznie nie powinno się zdarzyć)
        return max(tau, 0.5)
    else:
        raise ValueError(f"Unknown cutoff method: {cutoff}")

def effective_sample_size(n_samples: int, tau_int: float) -> float:
    """
    ESS = N / (2 * tau_int)
    Zabezpieczenia: tau_int <= 0 zwraca n_samples.
    """
    if tau_int <= 0:
        return float(n_samples)
    return float(n_samples) / (2.0 * float(tau_int))

def analyze_timeseries(x: np.ndarray, max_lag: Optional[int] = None) -> dict:
    """
    Pełna analiza: zwraca autokorelację, tau_int i ESS.
    """
    rho = autocorrelation(x, max_lag=max_lag)
    tau_int = integrated_autocorrelation_time(rho, cutoff="first_negative")
    ess = effective_sample_size(len(x), tau_int)
    return {"autocorrelation": rho, "tau_int": tau_int, "ESS": ess}

# Przykład użycia
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    res = analyze_timeseries(x, max_lag=100)
    print("tau_int:", res["tau_int"])
    print("ESS:", res["ESS"])
    print("rho[:10]:", res["autocorrelation"][:10])

