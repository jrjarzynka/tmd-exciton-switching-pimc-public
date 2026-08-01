"""
Test 4: artefakt periodycznosci GridPotential2D -- przed/po poprawce.

Rekonstruuje STARA (bledna) wersje interpolacji bezposrednio (bez potrzeby
historii gita) na podstawie komentarza w kodzie:

    # FIX: sasiad dla ostatniej komorki siatki powinien zawijac sie do
    # indeksu 0 (to JEST obraz periodyczny), a nie byc przycinany do nx-2.
    # Stare przycinanie po cichu uzywalo nieaktualnych danych (ix-1, ix)
    # dla kazdego periodycznego zawiniecia.

Test buduje prosty periodyczny landscape (MoirePotential zrasteryzowany na
grid), oblicza wartosci V(r) obiema metodami wzdluz linii przechodzacej
przez granice periodycznej komorki, i pokazuje nieciaglosc "przed" oraz jej
zniknieciem "po".
"""
import numpy as np
from tmd_pimc import MoirePotential, GridPotential2D

PERIOD_NM = 20.0
AMPLITUDE_EV = 0.020
GRID_N = 60   # celowo rzadka siatka, zeby artefakt byl dobrze widoczny

moire = MoirePotential(amplitude_eV=AMPLITUDE_EV, period_nm=PERIOD_NM)

# Budujemy grid TAK jak generate_potential_map.py: endpoint=False
x = np.linspace(-PERIOD_NM/2, PERIOD_NM/2, GRID_N, endpoint=False)
y = np.linspace(-PERIOD_NM/2, PERIOD_NM/2, GRID_N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")
pts = np.stack([X.ravel(), Y.ravel()], axis=1)
V = moire.value(pts).reshape(GRID_N, GRID_N)

grid_new = GridPotential2D(x_nm=x, y_nm=y, V_eV=V, periodic=True,
                            subtract_minimum=False)

def value_old_buggy(x_nm, y_nm, dx, dy, nx, ny, V):
    """Reprodukcja STAREGO zachowania: sasiad przycinany do nx-2/ny-2
    zamiast zawijany modulo nx/ny."""
    xmin, ymin = x_nm[0], y_nm[0]
    fx = (np.asarray(x_nm) - xmin) / dx  # placeholder, patrz nizej
    raise NotImplementedError

# Prostszy, rownowazny sposob: przeliczamy recznie wzdluz linii y=0,
# przechodzacej przez x = +period/2 (granica periodycznej komorki),
# porownujac NOWA metode (grid_new.value, poprawna) z RECZNA rekonstrukcja
# STAREJ (przycinanie zamiast zawijania) na tych samych danych V.

dx = float(x[1] - x[0])
dy = float(y[1] - y[0])
nx, ny = GRID_N, GRID_N

def old_buggy_lookup(xq, yq):
    ux = (xq - x[0]) / dx
    uy = (yq - y[0]) / dy
    ix = int(np.floor(ux)) % nx
    iy = int(np.floor(uy)) % ny
    wx = ux - np.floor(ux)
    wy = uy - np.floor(uy)
    # STARY (bledny) sposob: przycinanie zamiast zawijania modulo
    ix1 = min(ix + 1, nx - 1)
    iy1 = min(iy + 1, ny - 1)
    v00, v10 = V[ix, iy], V[ix1, iy]
    v01, v11 = V[ix, iy1], V[ix1, iy1]
    return (1-wx)*(1-wy)*v00 + wx*(1-wy)*v10 + (1-wx)*wy*v01 + wx*wy*v11

# Skan wzdluz linii przechodzacej przez granice komorki (x w poblizu +period/2)
scan_x = np.linspace(8.5, 11.5, 300)  # przekracza granice periodyczna przy x=10
scan_y0 = 0.0

v_new = np.array([grid_new.value(np.array([[xx, scan_y0]]))[0] for xx in scan_x])
v_old = np.array([old_buggy_lookup(xx, scan_y0) for xx in scan_x])

print("Maksymalna nieciaglosc (stara metoda, w poblizu granicy periodycznej):")
d_old = np.max(np.abs(np.diff(v_old)))
d_new = np.max(np.abs(np.diff(v_new)))
print(f"  stara: max|dV| miedzy sasiednimi punktami skanu = {d_old*1000:.3f} meV")
print(f"  nowa:  max|dV| miedzy sasiednimi punktami skanu = {d_new*1000:.3f} meV")
print(f"  stosunek: {d_old/d_new:.1f}x wieksza nieciaglosc w starej metodzie")

import csv
with open("/mnt/user-data/outputs/test4_periodicity_scan.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["x_nm", "V_old_buggy_eV", "V_new_fixed_eV"])
    for xx, vo, vn in zip(scan_x, v_old, v_new):
        w.writerow([xx, vo, vn])
print("\nZapisano test4_periodicity_scan.csv -- wykres V(x) obiema metodami,")
print("z pionowa linia przy x=10 (granica periodycznej komorki), pokaze")
print("widoczny skok w 'starej' krzywej i jego brak w 'nowej'.")
