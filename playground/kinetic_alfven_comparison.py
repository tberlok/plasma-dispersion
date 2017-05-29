from vlasov import Species, Electrons, VlasovFluid, Vlasov
from vlasov.cython.types import Complex
import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import newton
from vlasov import newton

tol = 1e-12

# Number density of ions
n0 = 1

# Magnetic field strength
B0 = 1.


Te = 1e-4

# Ion species
ion = Species(Tpar=1/2, Tperp=1/2, n0=n0)
species = (ion,)

# Electron fluid properties
electrons = Electrons(Te=Te)

# Initialize solver
vlasov = VlasovFluid(B0, species, electrons)

Nk = 800
kvec = np.logspace(-1, 0.8, Nk)

# Inclination
theta = 85*np.pi/180

plt.figure(1)
plt.clf()
fig, axes = plt.subplots(nrows=2, num=1)
omega = np.zeros(Nk, dtype=Complex)

omega[0] = 1e-2 - 1e-4j
kpar = kvec[0]*np.cos(theta)
kperp = kvec[0]*np.sin(theta)
omega[0] = newton(vlasov.det, omega[0], args=(kpar, kperp), tol=tol)

for i in range(1, Nk):
    kpar = kvec[i]*np.cos(theta)
    kperp = kvec[i]*np.sin(theta)
    omega[i] = newton(vlasov.det, omega[i-1], args=(kpar, kperp),
                      maxiter=1000)
    temp = vlasov.det(omega[i], kpar, kperp)
    assert np.abs(temp) < 10*tol, np.abs(temp)

axes[1].loglog(kvec, omega.real, 'C0-')
axes[0].loglog(kvec, -omega.imag, 'C0-', label=r"Vlasov-Fluid")

##############################################################################
# Full Vlasov solution

# species
ions = Species(Tpar=1/2, Tperp=1/2)
electrons = Species(Tpar=Te, Tperp=Te, mass=5.4e-3, charge=-1)
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
    omega[i] = newton(vlasov.det, omega[i-1], args=(kpar, kperp),
                      maxiter=1000)
    temp = vlasov.det(omega[i], kpar, kperp)
    assert np.abs(temp) < 10000*tol, np.abs(temp)

axes[1].loglog(kvec, omega.real, 'C1--')
axes[0].loglog(kvec, -omega.imag, 'C1--', label=r"Vlasov-Vlasov")
plt.xlabel(r"$k_\parallel$")
axes[1].set_ylabel(r"$Re(\omega/\Omega)$")
axes[0].set_ylabel(r"$-Im(\omega/\Omega)$")
axes[0].legend(frameon=False, loc='upper left')
print("Maximum growth rate is:", omega.imag.max())
plt.show()
