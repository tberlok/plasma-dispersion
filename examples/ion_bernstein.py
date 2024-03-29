from vlasov import Species, Vlasov
from vlasov.cython.types import Complex
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

# Magnetic field strength
B0 = 1.

# Species
ions = Species(Tpar=1, Tperp=1, n0=1)
electrons = Species(Tpar=1e8, Tperp=1e8, n0=1, mass=1/1836, charge=-1)
species = (ions, electrons)

# Initialize solver
vlasov = Vlasov(B0, species)

Nk = 1000
kmax = 10
kperp = np.linspace(kmax, 0.2, Nk)
kpar = 1e-8

omega = np.zeros(Nk, dtype=Complex)

# Guess for first solution
omega[0] = 1 + 1e-8j

# Get the first root
omega[0] = newton(vlasov.det, omega[0], args=(kpar, kperp[0]))

for i in range(1, Nk):
    omega[i] = newton(vlasov.det, omega[i-1], args=(kpar, kperp[i]))

# Plotting
fig, axes = plt.subplots(nrows=2)
axes[1].plot(kperp, omega.real, '-')
axes[0].plot(kperp, omega.imag, '-')
axes[1].set_ylabel(r"$Re(\omega/\Omega)$")
axes[0].set_ylabel(r"$Im(\omega/\Omega)$")
plt.xlim(0, 5)
plt.xlabel(r"$k_\perp$")
plt.savefig("../figures/full_vlasov_ion_bernstein.eps")
# plt.show()
