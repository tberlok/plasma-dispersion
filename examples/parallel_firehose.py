from vlasov import Species, Electrons, VlasovFluid
from vlasov.cython.types import Complex
import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import newton
from vlasov import newton

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

Nk = 400
kpar = np.linspace(0.2, 0.8, Nk)
kperp = 1e-16

omega = np.zeros(Nk, dtype=Complex)

# Guess for first solution
omega[0] = 0.4 + 0.04j

# Get the first root
omega[0] = newton(vlasov.det, omega[0], fprime=vlasov.derivative_det, args=(kpar[0], kperp))

for i in range(1, Nk):
    omega[i] = newton(vlasov.det, omega[i-1], fprime=vlasov.derivative_det, args=(kpar[i], kperp))

# Plotting
plt.figure(1)
plt.clf()
fig, axes = plt.subplots(nrows=2)
axes[1].plot(kpar, omega.real, '-')
axes[0].plot(kpar, omega.imag, '-')
axes[1].set_ylabel(r"$Re(\omega/\Omega)$")
axes[0].set_ylabel(r"$Im(\omega/\Omega)$")
plt.xlabel(r"$k_\parallel v_a/\Omega$")
# plt.savefig("../figures/parallel_firehose.eps")
print("Maximum growth rate is:", omega.imag.max())
plt.show()
