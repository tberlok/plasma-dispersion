from vlasov import Species, Electrons, VlasovFluid
from vlasov.cython.types import Complex
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

# Magnetic field strength
B0 = 1.
# Ion species
ion = Species(Tpar=1/2, Tperp=2)
species = (ion,)
# Electron fluid properties
electrons = Electrons(Te=0)

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
    axes[1].plot(kpar, omega.real, '-')
    axes[0].plot(kpar, omega.imag, '-')
axes[1].set_ylabel(r"$Re(\omega/\Omega)$")
axes[0].set_ylabel(r"$Im(\omega/\Omega)$")
plt.xlim(0, 2)
plt.xlabel(r"$k_\parallel v_a/\Omega$")
plt.savefig("../figures/ion_cyclotron.eps")
print("Maximum growth rate is:", omega.imag.max())
