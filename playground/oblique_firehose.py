from vlasov import Species, Electrons, VlasovFluid, Vlasov
from vlasov.cython.types import Complex
import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import newton
from vlasov import newton

tol = 1e-12
maxiter = 1000

# Number density of ions
n0 = 1

# Magnetic field strength
B0 = 1.

# Parallel and perpendicular beta (as in the hydros paper)
beta_para = 1
beta_perp = 4.

mu0 = 1

Te = 1e-4

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

kmax = 0.2
Nk = 400

# 45 degree inclination
theta = 45*np.pi/180

plt.figure(1)
plt.clf()
fig, axes = plt.subplots(nrows=2, num=1)
kvecs = (np.linspace(kmax, 0.01, Nk), np.linspace(kmax, 5.0*kmax, Nk))
for kvec in kvecs:
    omega = np.zeros(Nk, dtype=Complex)

    # Guess needs to be almost purely growing and quite close to solution...
    omega[0] = 1e-8 + 0.035j
    kpar = kvec[0]*np.cos(theta)
    kperp = kvec[0]*np.sin(theta)
    omega[0] = newton(vlasov.det, omega[0], args=(kpar, kperp), tol=tol)

    for i in range(1, Nk):
        kpar = kvec[i]*np.cos(theta)
        kperp = kvec[i]*np.sin(theta)
        omega[i] = newton(vlasov.det, omega[i-1], args=(kpar, kperp), tol=tol,
                          maxiter=maxiter)
        temp = vlasov.det(omega[i], kpar, kperp)
        assert np.abs(temp) < 100*tol, np.abs(temp)

    axes[1].plot(kvec, omega.real, 'C0-')
    axes[0].plot(kvec, omega.imag, 'C0-')
axes[1].set_ylabel(r"$Re(\omega/\Omega)$")
axes[0].set_ylabel(r"$Im(\omega/\Omega)$")

##############################################################################
# Full Vlasov solution

# species
ions = Species(Tpar=Tpar, Tperp=Tperp)
electrons = Species(Tpar=Te, Tperp=Te, mass=5.4e-3, charge=-1)
species = (ions, electrons)


# Initialize solver
vlasov = Vlasov(B0, species)

kvecs = (np.linspace(kmax, 0.01, Nk), np.linspace(kmax, 5.0*kmax, Nk))
for kvec in kvecs:
    omega = np.zeros(Nk, dtype=Complex)

    # Guess needs to be almost purely growing and quite close to solution...
    omega[0] = 1e-8 + 0.035j
    kpar = kvec[0]*np.cos(theta)
    kperp = kvec[0]*np.sin(theta)
    omega[0] = newton(vlasov.det, omega[0], args=(kpar, kperp), tol=tol)

    for i in range(1, Nk):
        kpar = kvec[i]*np.cos(theta)
        kperp = kvec[i]*np.sin(theta)
        omega[i] = newton(vlasov.det, omega[i-1], args=(kpar, kperp), tol=tol,
                          maxiter=maxiter)
        temp = vlasov.det(omega[i], kpar, kperp)
        assert np.abs(temp) < 100*tol, np.abs(temp)

    axes[1].plot(kvec, omega.real, 'C1--')
    axes[0].plot(kvec, omega.imag, 'C1--')

axes[1].set_ylabel(r"$Re(\omega/\Omega)$")
axes[0].set_ylabel(r"$-Im(\omega/\Omega)$")
axes[0].legend(frameon=False, loc='upper left')

plt.show()
plt.savefig("../figures/oblique_firehose.eps")
