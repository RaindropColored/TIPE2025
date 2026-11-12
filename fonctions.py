import matplotlib.pyplot as plt
import numpy as np

# Visualisation
offset = 1.
def visualiser(tableau_de_graphes, abcisses, temps, methode) : 
    for i, ui in enumerate(tableau_de_graphes) :
        plt.plot(abcisses, ui + offset * i, label=f"t = {temps[i]:.3f}")
    plt.title(f"Propagation d'une gaussienne avec {methode}")
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.legend()
    plt.grid()
    plt.show()



#solitons

# Fonction sech (secante hyperbolique)
def sech(x):
    return 1.0 / np.cosh(x)

# Condition initiale : soliton analytique
def soliton(celerite, pos_initiale, x) : 
    return 0.5* celerite * sech(0.5 * np.sqrt(celerite) * (x - pos_initiale))**2

# condition initiale : deux solitons 
def deux_solitons(celerite_1, celerite_2, pos_initiale_1, pos_initiale_2, x) : 
    return ((0.5* celerite_1 * sech(0.5 * np.sqrt(celerite_1) * (x - pos_initiale_1))**2) 
            + (0.5* (celerite_2) * sech(0.5 * np.sqrt(celerite_2) * (x - pos_initiale_2))**2))

# condition initiale : gaussienne 
def gaussienne(amplitude, largeur, pos_centre, x):
    return amplitude * np.exp(-((x - pos_centre)**2) / (2 * largeur**2))

# condition initiale : sinusoidale
def sinusoidale(amplitude, k, x):
    return amplitude * np.sin(k * x * 2 * np.pi)


#pour comparer aux résultats de la simuation
def soliton_exact(c, x0, x, t):
    return 0.5 * c * (1 / np.cosh(0.5 * np.sqrt(c) * (x - c * t - x0)))**2



# fonctions utiles

def nsteps(tmax, dt) :
    return int(tmax / dt)  # Nombre d'étapes de temps
def save_interval(tmax, dt, nb_snapshots = 9) :
    return int(nsteps(tmax, dt)/nb_snapshots) # Nombre d'itérations entre snapshots



# différences finies

# fonction pour la dérivée première (périodique)
def d_dx(u, dx):
    # u[i+1] - u[i-1] / (2*dx), avec périodicité
    return (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)

def d3_dx3_ordre_2(u, dx):
    return ((
        -np.roll(u, 2) 
        + 2*np.roll(u, 1) 
        - 2*np.roll(u, -1) 
        + np.roll(u, -2)) 
        / (2 * dx**3))

# fonction pour la dérivée troisième (périodique) différences finies ordre 4 
def d3_dx3_ordre_4(u, dx):
    return (
        (-1.0 * np.roll(u, -3)
         + 8.0 * np.roll(u, -2)
         - 13.0 * np.roll(u, -1)
         + 13.0 * np.roll(u, 1)
         - 8.0 * np.roll(u, 2)
         + 1.0 * np.roll(u, 3))
        / (8 * dx**3)
    )

# on choisit l'approximation spatiale.
def d3_dx3(u, dx):
    return d3_dx3_ordre_2(u, dx)


# membre de droite de KdV : -6u u_x - u_xxx
def membre_de_droite_kdv(u, dx):
    return - 6 * u * d_dx(u, dx) - d3_dx3(u, dx)

# membre de droite linéaire : - u_xxx
def membre_de_droite_lineaire(u, dx):
    return - d3_dx3(u, dx)

# membre de droite non linéaire : -u u_x
def membre_de_droite_non_lineaire(u, dx):
    return - u * d_dx(u, dx) 



# méthode de Runge-Kutta d'ordre 4 
def rk4_step(u, dx, dt):
    k1 = membre_de_droite_kdv(u, dx)
    k2 = membre_de_droite_kdv(u + 0.5 * dt * k1, dx)
    k3 = membre_de_droite_kdv(u + 0.5 * dt * k2, dx)
    k4 = membre_de_droite_kdv(u + dt * k3, dx)
    return u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def linear_step(u, dx, dt):
    k1 = membre_de_droite_lineaire(u, dx)
    k2 = membre_de_droite_lineaire(u + 0.5 * dt * k1, dx)
    k3 = membre_de_droite_lineaire(u + 0.5 * dt * k2, dx)
    k4 = membre_de_droite_lineaire(u + dt * k3, dx)
    return u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def non_linear_step(u, dx, dt):
    k1 = membre_de_droite_non_lineaire(u, dx)
    k2 = membre_de_droite_non_lineaire(u + 0.5 * dt * k1, dx)
    k3 = membre_de_droite_non_lineaire(u + 0.5 * dt * k2, dx)
    k4 = membre_de_droite_non_lineaire(u + dt * k3, dx)
    return u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# boucles d'intégration en temps
def simuler_rk4(u, dx, dt, tmax, snapshots, times):
    for n in range(1, nsteps(tmax, dt) + 1):
        u = rk4_step(u, dx, dt)
        if n % save_interval(tmax, dt) == 0:
            snapshots.append(u.copy())
            times.append(n * dt)        
            delta = np.linalg.norm(u - snapshots[-2])
            print(f"Étape {n}, variation de u : {delta:.5e}")

def simuler_lineaire(u, dx, dt, tmax, snapshots, times):
    for n in range(1, nsteps(tmax, dt) + 1):
        u = linear_step(u, dx, dt)
        if n % save_interval(tmax, dt) == 0:
            snapshots.append(u.copy())
            times.append(n * dt)        
            delta = np.linalg.norm(u - snapshots[-2])
            print(f"Étape {n}, variation de u : {delta:.5e}")

def simuler_non_lineaire(u, dx, dt, tmax, snapshots, times):
    for n in range(1, nsteps(tmax, dt) + 1):
        u = non_linear_step(u, dx, dt)
        if n % save_interval(tmax, dt) == 0:
            snapshots.append(u.copy())
            times.append(n * dt)        
            delta = np.linalg.norm(u - snapshots[-2])
            print(f"Étape {n}, variation de u : {delta:.5e}")



# méthode leapfrog

def init_leapfrog(condition_initiale, dt, dx) : 
    return condition_initiale + dt * membre_de_droite_kdv(condition_initiale, dx)

def leapfrog_step(u_nm1, u_n, dx, dt):
    membre_de_droite = membre_de_droite_kdv(u_n, dx)
    return u_nm1 + 2 * dt * membre_de_droite

# Boucle de simulation
def simuler_leapfrog_zk(u_nm1, u_n, dx, dt, tmax, snapshots, times):
    for n in range(nsteps(tmax, dt)):

        u_np1 = leapfrog_step(u_nm1, u_n, dx, dt)

        u_nm1 = u_n.copy()
        u_n = u_np1.copy()

        if n % save_interval(tmax, dt) == 0:
            snapshots.append(u_n.copy())
            times.append((n + 2)* dt)
    return u_n



# méthode d'euler

def euler_step(u_n, dx, dt):
    membre_de_droite = membre_de_droite_kdv(u_n, dx)
    return u_n + dt * membre_de_droite

# boucle de simulation
def simuler_euler(u_n, dx, dt, tmax, snapshots, times):
    for n in range(nsteps(tmax, dt)):

        u_np1 = euler_step(u_n, dx, dt)

        u_n = u_np1.copy()

        if n % save_interval(tmax, dt)  == 0:
            snapshots.append(u_n.copy())
            times.append((n + 1) * dt)


