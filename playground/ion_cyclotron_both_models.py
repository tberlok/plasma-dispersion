from vlasov import Species, Electrons, VlasovFluid, Vlasov
from vlasov.cython.types import Complex
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

# Magnetic field strength
B0 = 1.

##############################################################################
# Vlasov-Fluid solution

# Ion species
ion = Species(Tpar=1/2, Tperp=2)
species = (ion,)
# Electron fluid properties
electrons = Electrons(Te=1/2)

# Initialize solver
vlasov = VlasovFluid(B0, species, electrons)

Nk = 400
kmax = 0.7207846846846846

kvecs = (np.linspace(kmax, 1e-4, Nk), np.linspace(kmax, 2, Nk))
kperp = 1e-8

omega = np.zeros(Nk, dtype=Complex)

fig, axes = plt.subplots(nrows=2)
for kpar in kvecs:
    # Guess for first solution
    omega[0] = 0.68533380775977581+0.27820200887704144j

    # Get the first root
    omega[0] = newton(vlasov.det, omega[0], args=(kpar[0], kperp))

    for i in range(1, Nk):
        omega[i] = newton(vlasov.det, omega[i-1], args=(kpar[i], kperp))

    # Plotting
    axes[1].plot(kpar, omega.real, 'b-')
    axes[0].plot(kpar, omega.imag, 'b-')
axes[1].set_ylabel(r"$Re(\omega/\Omega)$")
axes[0].set_ylabel(r"$Im(\omega/\Omega)$")
plt.xlim(0, 2)
print("Maximum growth rate is:", omega.imag.max())

##############################################################################
# Full Vlasov solution

# species
ions = Species(Tpar=1/2, Tperp=2)
electrons = Species(Tpar=1/2, Tperp=1/2, mass=5.4e-3, charge=-1)
species = (ions, electrons)


# Initialize solver
vlasov = Vlasov(B0, species)

omega = np.zeros(Nk, dtype=Complex)

for kpar in kvecs:
    # Guess for first solution
    omega[0] = 0.68533380775977581+0.27820200887704144j

    # Get the first root
    omega[0] = newton(vlasov.det, omega[0], args=(kpar[0], kperp))

    for i in range(1, Nk):
        omega[i] = newton(vlasov.det, omega[i-1], args=(kpar[i], kperp))

    # Plotting
    axes[1].plot(kpar, omega.real, 'r--')
    axes[0].plot(kpar, omega.imag, 'r--')
axes[1].set_ylabel(r"$Re(\omega/\Omega)$")
axes[0].set_ylabel(r"$Im(\omega/\Omega)$")
plt.xlabel(r"$k_\parallel$")
print("Maximum growth rate is:", omega.imag.max())
plt.show()
