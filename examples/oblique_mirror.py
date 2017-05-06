from vlasov import Species, Electrons, VlasovFluid
from vlasov.cython.types import Complex
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

# Number density of ions
n0 = 1

# Magnetic field strength
B0 = 1.

# Parallel and perpendicular beta (as in the hydros paper)
beta_para = 6.
beta_perp = 10

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

Nk = 400
kpar = np.linspace(1e-2, 2, Nk)
kperp = 1e-8

kmax = 0.2
Nk = 400

# 45 degree inclination
theta = 45*np.pi/180

fig, axes = plt.subplots(nrows=2, num=1)
kvecs = (np.linspace(kmax, 1e-4, Nk), np.linspace(kmax, 2.5*kmax, Nk))
for kvec in kvecs:
    omega = np.zeros(Nk, dtype=Complex)

    # Guess needs to be almost purely growing and quite close to solution...
    omega[0] = 1e-4 + 0.027820200887704144j
    kpar = kvec[0]*np.cos(theta)
    kperp = kvec[0]*np.sin(theta)
    omega[0] = newton(vlasov.det, omega[0], args=(kpar, kperp))

    for i in range(1, Nk):
        kpar = kvec[i]*np.cos(theta)
        kperp = kvec[i]*np.sin(theta)
        omega[i] = newton(vlasov.det, omega[i-1], args=(kpar, kperp))

    axes[1].plot(kvec, omega.real, '-')
    axes[0].plot(kvec, omega.imag, '-')
axes[1].set_ylabel(r"$Re(\omega/\Omega)$")
axes[0].set_ylabel(r"$Im(\omega/\Omega)$")
plt.savefig("../figures/oblique_mirror.eps")
