from vlasov import Species, Electrons, VlasovFluid, Vlasov
from vlasov.cython.types import Complex
import numpy as np
import matplotlib.pyplot as plt
from vlasov import newton

# This example shows how the oblique mirror instability has the same growth
# rate using the Vlasov-Fluid model and the full Vlasov model in the limit
# Te << 1. This seems

# Number density of ions
n0 = 1

# Magnetic field strength
B0 = 1.

# Parallel and perpendicular beta (as in the hydros paper)
beta_para = 1
beta_perp = 4

Te = 0.5

tol = 1e-12
maxiter = 1000

mu0 = 1

# Convert to temperature using beta_para = 2*Tpar/mass /(B0**2)*mu*n0*mass
Tpar = beta_para/2*mu0*n0*B0**2
Tperp = beta_perp/2*mu0*n0*B0**2

# Ion species
ion = Species(Tpar=Tpar, Tperp=Tperp, n0=n0)
species = (ion,)
# Electron fluid properties
electrons = Electrons(Te=Te)

# Initialize solver
vlasov = VlasovFluid(B0, species, electrons)

Nk = 400
kpar = np.linspace(1e-2, 2, Nk)
kperp = 1e-8

kmax = 0.2
Nk = 400

# 45 degree inclination
theta = 45*np.pi/180
plt.figure(1)
plt.clf()
first = True
fig, axes = plt.subplots(nrows=2, num=1)
kvecs = (np.linspace(kmax, 1e-4, Nk), np.linspace(kmax, 5*kmax, Nk))
for kvec in kvecs:
    omega = np.zeros(Nk, dtype=Complex)

    # Guess needs to be almost purely growing and quite close to solution...
    omega[0] = 1e-4 + 0.15j
    kpar = kvec[0]*np.cos(theta)
    kperp = kvec[0]*np.sin(theta)
    omega[0] = newton(vlasov.det, omega[0], args=(kpar, kperp), tol=tol,
                      maxiter=maxiter)

    for i in range(1, Nk):
        kpar = kvec[i]*np.cos(theta)
        kperp = kvec[i]*np.sin(theta)
        omega[i] = newton(vlasov.det, omega[i-1], args=(kpar, kperp), tol=tol,
                          maxiter=maxiter)
        assert(np.abs(vlasov.det(omega[i], kpar, kperp)) < 100*tol)

    axes[1].plot(kvec, omega.real, 'C0-')
    if first:
        axes[0].plot(kvec, omega.imag, 'C0-', label=r"Vlasov-Fluid")
        first = False
    else:
        axes[0].plot(kvec, omega.imag, 'C0-')
    print("Maximum growth rate is:", omega.imag.max())
    index = np.argmax(omega.imag)
    print('Maximally unstable wavelength is', 2*np.pi/kvec[index])

##############################################################################
# Full Vlasov solution

# species
ions = Species(Tpar=Tpar, Tperp=Tperp, n0=n0)
electrons = Species(Tpar=Te, Tperp=Te, mass=5.4e-3, charge=-1)
species = (ions, electrons)


# Initialize solver
vlasov = Vlasov(B0, species)
first = True
for kvec in kvecs:
    omega = np.zeros(Nk, dtype=Complex)

    # Guess needs to be almost purely growing and quite close to solution...
    omega[0] = 1e-4 + 0.15j
    kpar = kvec[0]*np.cos(theta)
    kperp = kvec[0]*np.sin(theta)
    omega[0] = newton(vlasov.det, omega[0], args=(kpar, kperp),
                      maxiter=maxiter)

    for i in range(1, Nk):
        kpar = kvec[i]*np.cos(theta)
        kperp = kvec[i]*np.sin(theta)
        omega[i] = newton(vlasov.det, omega[i-1], args=(kpar, kperp),
                          maxiter=maxiter, tol=tol)
        assert(np.abs(vlasov.det(omega[i], kpar, kperp)) < 100*tol)

    axes[1].plot(kvec, omega.real, 'C1--')
    if first:
        axes[0].plot(kvec, omega.imag, 'C1--', label=r"Vlasov-Vlasov")
        first = False
    else:
        axes[0].plot(kvec, omega.imag, 'C1--')
    print("Maximum growth rate is:", omega.imag.max())
    index = np.argmax(omega.imag)
    print('Maximally unstable wavelength is', 2*np.pi/kvec[index])
axes[1].set_ylabel(r"$Re(\omega/\Omega)$")
axes[0].set_ylabel(r"$Im(\omega/\Omega)$")
axes[0].legend(frameon=False, loc='upper left')

axes[1].set_ylim(-0.25, 0.25)
plt.show()
plt.savefig("../figures/oblique_mirror.eps")
