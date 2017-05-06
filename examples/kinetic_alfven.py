from vlasov import Species, Electrons, VlasovFluid
from vlasov.cython.types import Complex
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

# Number density of ions
n0 = 1

# Magnetic field strength
B0 = 1.

# Ion species
ion = Species(Tpar=1/2, Tperp=1/2, n0=n0)
species = (ion,)

# Electron fluid properties
electrons = Electrons(Te=1/2)

# Initialize solver
vlasov = VlasovFluid(B0, species, electrons)

Nk = 400
kvec = np.logspace(-1, 1, Nk)

# Inclination
theta = 85*np.pi/180

fig, axes = plt.subplots(nrows=2, num=1)
omega = np.zeros(Nk, dtype=Complex)

omega[0] = 1e-2 - 1e-4j
kpar = kvec[0]*np.cos(theta)
kperp = kvec[0]*np.sin(theta)
omega[0] = newton(vlasov.det, omega[0], args=(kpar, kperp))

for i in range(1, Nk):
    kpar = kvec[i]*np.cos(theta)
    kperp = kvec[i]*np.sin(theta)
    omega[i] = newton(vlasov.det, omega[i-1], args=(kpar, kperp))

axes[1].loglog(kvec, omega.real, '-')
axes[0].loglog(kvec, -omega.imag, '-')
axes[1].set_ylabel(r"$Re(\omega/\Omega)$")
axes[0].set_ylabel(r"$-Im(\omega/\Omega)$")
plt.savefig("../figures/kinetic_alfven.eps")
