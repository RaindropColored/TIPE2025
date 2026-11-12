import numpy as np
import matplotlib.pyplot as plt
from fonctions import *



# test convergence avec critere CFL

L = 40.0
c = 1.0
x0 = L / 2
tmax = 1.0
CFL = 0.1  # Facteur de proportion entre dt et dx (stabilité + précision)

Ns = [500, 1000, 2000, 3000]  # Raffinement progressif
errors = []
dxs = []

for N in Ns:
    x = np.linspace(0, L, N, endpoint=False)
    dx = L / N
    dt = (CFL * dx**3) / (0.5 * c) # critere CFL
    dxs.append(dx)

    def sol_ref(t):
        return soliton_exact(c, x0, x, t)

    # Initialisation
    u_m1 = soliton(c, x0, x)
    u_0 = init_leapfrog(u_m1, dt, dx)
    snapshots = [u_0.copy()]
    times = [dt]

    # Simulation
    simuler_leapfrog_zk(u_m1, u_0, dx, dt, tmax, snapshots, times)
    sol_num = snapshots[-1]

    # Erreur L2
    error = np.linalg.norm(sol_num - sol_ref(times[-1]), ord=2) * np.sqrt(dx)
    errors.append(error)
    print(f"N = {N:5d} | dx = {dx:.2e} | dt = {dt:.2e} → erreur L2 = {error:.2e}")

# Affichage
plt.loglog(dxs, errors, 'o-', label='Erreur couplée dt ~ dx')

plt.xlabel("dx (avec dt ∝ dx)")
plt.ylabel("Erreur L2")
plt.grid(True, which='both')
plt.legend()
plt.title("Convergence couplée du schéma Leapfrog (dt ∝ dx)")
plt.show()

plt.clf()

# test convergence selon dx

L = 40.0
c = 1.0
x0 = L / 2
tmax = 1.0
dt = 2e-6 

# résolution en espace
Ns = [50, 250, 500, 1000]
errors = []

for N in Ns:
    x = np.linspace(0, L, N, endpoint=False)
    dx = L / N
    
    # Solution exacte
    def sol_ref(t):
        return soliton_exact(c, x0, x, t)

    # leapfrog
    u_m1 = soliton(c, x0, x)
    u_0 = init_leapfrog(u_m1, dt, dx)
    snapshots = [u_0.copy()]
    times = [dt]

    simuler_leapfrog_zk(u_m1, u_0, dx, dt, tmax, snapshots, times)
    sol_num = snapshots[-1]
    
    # erreur L2
    error = np.linalg.norm(sol_num - sol_ref(times[-1]), ord=2) * np.sqrt(dx)
    errors.append(error)
    print(f"N = {N:5d} → dx = {dx:.5e} → erreur L2 = {error:.2e}")

dxs = [L / N for N in Ns]
plt.loglog(dxs, errors, 'o-', label='Erreur')

plt.xlabel("dx")
plt.ylabel("Erreur L2")
plt.grid(True, which='both')
plt.legend()
plt.title("Convergence en espace du schéma Leapfrog")
plt.show()

plt.clf()

# test convergence selon dt

L = 40.0
N = 500
x = np.linspace(0, L, N, endpoint=False)
dx = L / N
c = 5.0
x0 = L / 2
tmax = 1.0

# Référence théorique
def sol_ref(t) : 
    return soliton_exact(c, x0, x, t)

dts = [
    1e-2,
    1e-3,
    1e-4,
    2e-5,   
    1e-5,
    5e-6,
    3e-6,
    2e-6,   
    1e-6
]
errors = []

for dt in dts:
    u_m1 = soliton(c, x0, x)
    u_0 = init_leapfrog(u_m1, dt, dx)
    snapshots = [u_0.copy()]
    times = [dt]
    simuler_leapfrog_zk(u_m1, u_0, dx, dt, tmax, snapshots, times)
    sol_num = snapshots[-1]
    sol_ref_last = sol_ref(times[-1])
    error = np.linalg.norm(sol_num - sol_ref_last, ord = 2) * np.sqrt(dx)
    print("times[-1] = ", times[-1])
    print(f"dt = {dt:.5e} → erreur L2 = {error:.2e}")
    plt.plot(x, sol_num)
    plt.plot(x, sol_ref_last )
    plt.plot(x, sol_num - sol_ref_last)
    plt.show()
    plt.clf()
    visualiser(snapshots, x, times, "ZK")
    plt.clf()
    errors.append(error)


plt.loglog(dts, errors, 'o-', label='Erreur')
plt.xlabel("dt")
plt.ylabel("Erreur L2")
plt.grid(True, which='both')
plt.legend()
plt.title(f"Convergence en temps du schéma Leapfrog pour N = {N}")
plt.show()