from fonctions import *
import numpy as np

# paramètres de simulation
L = 100.0           # Longueur du domaine
N = 400            # Nombre de points de grille
dx = L / N         # Pas d'espace
x = np.linspace(0, L, N, endpoint=False)  # Grille spatiale

# paramètres condition initiale
c = 1.0            # Vitesse du soliton
x0 = L / 2         # Position initiale
tmax = 10.0        # Temps final
dt = 0.0001          # Pas de temps


# simulation du terme non linéaire avec rk4

u = soliton(c, x0, x)

# (pour stocker les solutions à différents temps)
snapshots = [u.copy()]
times = [0.0]

simuler_non_lineaire(u, dx, dt, tmax, snapshots, times)

visualiser(snapshots, x, times, "Terme non linéaire")



# simulation du croisement de deux solitons avec rk4

u = deux_solitons(1.0, 2.0, L/2, L/3, x)

snapshots = [u.copy()]
times = [0.0]

simuler_rk4(u, dx, dt, tmax, snapshots, times)

visualiser(snapshots, x, times, "Runge-Kutta d'ordre 4")


# simulation d'une gaussienne avec zk

u_m1 = gaussienne(1.0, 1.0, L/2, x)
u_0 = init_leapfrog(u_m1, dt, dx)

snapshots = [u_0.copy()]
times = [dt]

simuler_leapfrog_zk(u_m1, u_0, dx, dt, tmax, snapshots, times)

visualiser(snapshots, x, times, "Zabusky-Kruskal")