from vlasov import Species, Electrons, VlasovFluid
from vlasov.cython.types import Complex
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

# Magnetic field strength
B0 = 1.
# Ion species
ion = Species(Tpar=2, Tperp=1/2, n0=1)
species = (ion,)
# Electron fluid properties
electrons = Electrons(Te=0)

# Initialize solver
vlasov = VlasovFluid(B0, species, electrons)

Nk = 400
kpar = np.linspace(1e-2, 2, Nk)
kperp = 1e-8

omega = np.zeros(Nk, dtype=Complex)

# Guess for first solution
omega[0] = 1 + 1j

# Get the first root
omega[0] = newton(vlasov.det, omega[0], args=(kpar[0], kperp))

for i in range(1, Nk):
    omega[i] = newton(vlasov.det, omega[i-1], args=(kpar[i], kperp))

# Plotting
fig, axes = plt.subplots(nrows=2)
axes[1].plot(kpar, omega.real, '-')
axes[0].plot(kpar, omega.imag, '-')
axes[1].set_ylabel(r"$Re(\omega/\Omega)$")
axes[0].set_ylabel(r"$Im(\omega/\Omega)$")
plt.xlim(0, 2)
plt.xlabel(r"$k_\parallel v_a/\Omega$")
plt.savefig("../figures/parallel_firehose.eps")
print("Maximum growth rate is:", omega.imag.max())
