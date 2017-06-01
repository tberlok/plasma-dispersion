from vlasov import Species, Electrons, VlasovFluid
from vlasov.cython.types import Complex
import numpy as np
import matplotlib.pyplot as plt
from vlasov import newton

tol = 1e-16

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
electrons = Electrons(Te=0.0)

# Initialize solver
vlasov = VlasovFluid(B0, species, electrons)

Nk = 400
kpar = np.linspace(1e-2, 2, Nk)
kperp = 1e-8

kmax = 0.4
Nk = 1000

# 45 degree inclination
theta = 53*np.pi/180
plt.figure(1)
plt.clf()
fig, axes = plt.subplots(nrows=2, num=1)
kvecs = (np.linspace(kmax, 0.3, Nk), np.linspace(kmax, 0.6, Nk))
for kvec in kvecs:
    omega = np.zeros(Nk, dtype=Complex)

    # Guess needs to be almost purely growing and quite close to solution...
    omega[0] = 1e-4 + 0.06j
    kpar = kvec[0]*np.cos(theta)
    kperp = kvec[0]*np.sin(theta)
    omega[0] = newton(vlasov.det, omega[0], args=(kpar, kperp))

    for i in range(1, Nk):
        kpar = kvec[i]*np.cos(theta)
        kperp = kvec[i]*np.sin(theta)
        omega[i] = newton(vlasov.det, omega[i-1], args=(kpar, kperp), tol=tol,
                          maxiter=1000)
        temp = vlasov.det(omega[i], kpar, kperp)
        assert abs(temp) <= 10*tol, abs(temp)

    axes[1].plot(kvec, omega.real, 'C0-')
    axes[0].plot(kvec, omega.imag, 'C0-')
axes[1].set_ylabel(r"$Re(\omega/\Omega)$")
axes[0].set_ylabel(r"$Im(\omega/\Omega)$")
print("Maximum growth rate is:", omega.imag.max())
index = np.argmax(omega.imag)
print("Most unstable wavelength is:", 2*np.pi/kvec[index])
# plt.savefig("../figures/oblique_firehose.eps")
plt.show()
