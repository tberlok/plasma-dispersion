from vlasov import Species, Electrons, VlasovFluid, Vlasov
from vlasov.cython.types import Complex
import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import newton
from vlasov import newton

tol = 1e-14

# Number density of ions
n0 = 1

# Magnetic field strength
B0 = 1.

# Ion species
ion = Species(Tpar=1/2, Tperp=1/2, n0=n0)
species = (ion,)

# Electron fluid properties
electrons = Electrons(Te=4)

# Initialize solver
vlasov = VlasovFluid(B0, species, electrons)

Nk = 400
kvec = np.logspace(-2, 1, Nk)

# Inclination
theta = 87.5*np.pi/180

fig, axes = plt.subplots(nrows=2, num=1)
omega = np.zeros(Nk, dtype=Complex)

omega[0] = 1e-2 - 1e-4j
kpar = kvec[0]*np.cos(theta)
kperp = kvec[0]*np.sin(theta)
omega[0] = newton(vlasov.det, omega[0], args=(kpar, kperp), tol=tol)

for i in range(1, Nk):
    kpar = kvec[i]*np.cos(theta)
    kperp = kvec[i]*np.sin(theta)
    omega[i] = newton(vlasov.det, omega[i-1], args=(kpar, kperp))
    temp = vlasov.det(omega[i], kpar, kperp)
    print(temp.imag - omega[i].imag, temp.real - omega[i].real)

axes[1].loglog(kvec, omega.real, 'b-')
axes[0].loglog(kvec, -omega.imag, 'b-', label=r"Vlasov-Fluid")

##############################################################################
# Full Vlasov solution

# species
ions = Species(Tpar=1/2, Tperp=1/2)
electrons = Species(Tpar=1/2, Tperp=1/2, mass=1e-12, charge=-1)
species = (ions, electrons)


# Initialize solver
vlasov = Vlasov(B0, species)

# omega[0] = 1e-2 - 1e-4j
kpar = kvec[0]*np.cos(theta)
kperp = kvec[0]*np.sin(theta)
omega[0] = newton(vlasov.det, omega[0], args=(kpar, kperp), tol=tol)

for i in range(1, Nk):
    kpar = kvec[i]*np.cos(theta)
    kperp = kvec[i]*np.sin(theta)
    omega[i] = newton(vlasov.det, omega[i-1], args=(kpar, kperp))
    temp = vlasov.det(omega[i], kpar, kperp)
    # print(temp.imag - omega[i].imag, temp.real - omega[i].real)

axes[1].loglog(kvec, omega.real, 'r--')
axes[0].loglog(kvec, -omega.imag, 'r--', label=r"Vlasov-Vlasov")
plt.xlabel(r"$k_\parallel$")
axes[1].set_ylabel(r"$Re(\omega/\Omega)$")
axes[0].set_ylabel(r"$-Im(\omega/\Omega)$")
axes[0].legend(frameon=False, loc='upper left')
print("Maximum growth rate is:", omega.imag.max())
plt.show()
