from vlasov import Species, Electrons, VlasovFluid, Solver
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from vlasov import make_guess

# Number density of ions
n0 = 1

# Magnetic field strength
B0 = 1.

# Parallel and perpendicular beta (as in the hydros paper)
beta_para = 2.8
beta_perp = 0.4*beta_para

mu0 = 1

# Convert to temperature using beta_para = 2*Tpar/mass /(B0**2)*mu*n0*mass
Tpar = beta_para/2*mu0*n0*B0**2
Tperp = beta_perp/2*mu0*n0*B0**2

# Ion species
ion = Species(Tpar=Tpar, Tperp=Tperp, n0=n0)
species = (ion,)
# Electron fluid properties
electrons = Electrons(Te=0)

# Initialize solver
vlasov = VlasovFluid(B0, species, electrons)

solver1 = Solver(vlasov, tol=1e-10, maxiter=1000)
solver2 = Solver(vlasov, tol=1e-10, maxiter=1000)

theta1 = np.linspace(0.01, 30, 90)*np.pi/180
theta2 = np.linspace(40, 8, 150)*np.pi/180
theta3 = np.linspace(40, 75, 105)*np.pi/180

N = 400

for theta in [theta2, theta3]:
    # Guess and its corresponding wavenumber
    for i in range(len(theta)):
        k0, guess = make_guess(theta[i], solver1)
        if k0 is None:
            k0 = 0.4
            guess = 1e-4 + 0.06j
        solver1(k0, 1.0, N, theta[i], guess)
        solver1(k0, 0.01, N, theta[i], guess)

for theta in [theta1, ]:
    # Guess and its corresponding wavenumber
    for i in range(len(theta)):
        k0, guess = make_guess(theta[i], solver2)
        if k0 is None:
            k0 = 0.2
            guess = 0.4 + 0.04j
        solver2(k0, 1.0, N, theta[i], guess)
        solver2(k0, 0.01, N, theta[i], guess)


data1 = solver1.create_data()
data2 = solver2.create_data()

data = np.concatenate((data1, data2))
index = data[:, 2].imag > 1e-3
data = data[index, :]

k = np.linspace(0.01, 1, 1000)
thetas = np.linspace(0.01, 75, 1000)*np.pi/180
k_g, theta_g = np.meshgrid(k, thetas)
omega = griddata((data[:, 0].real, data[:, 1].real), data[:, 2].imag,
                 (k_g, theta_g))
extent = [0.01, 1.0, 0.01, 75]
# im = plt.imshow(omega, origin='lower', extent=extent, aspect=1/200,
#                 vmin=0, vmax=0.06)
# CS = plt.contour(omega, colors='k', aspect=1/200, extent=extent,
#                  levels=[0.015, 0.025, 0.035, 0.045, 0.055])
CS = plt.contourf(omega, 20, aspect=1/200, extent=extent, cmap='rainbow')
plt.xlabel('$k$')
plt.ylabel(r'$\theta$')
# plt.clabel(CS, inline=1, fontsize=8)
# CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)
CBI = plt.colorbar(CS, orientation='horizontal', shrink=0.8)
plt.ylim(0, 70)
plt.xlim(0.2, 0.8)
plt.savefig('hellinger2000.pdf')
plt.show()

gam = []
thet = []
for m in range(solver1.M):
    gam.append(np.max(solver1.solutions[m]['omega'].imag))
    thet.append(solver1.solutions[m]['theta'])
