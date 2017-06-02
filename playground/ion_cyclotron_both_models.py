from vlasov import Species, Electrons, VlasovFluid, Vlasov, Solver
import numpy as np
import matplotlib.pyplot as plt

# Magnetic field strength
B0 = 1.

Tpar = 1/2
Tperp = 2
Te = 1/2

theta = 1e-4

Nk = 400
k0 = 0.7207846846846846
guess = 0.68533380775977581+0.27820200887704144j

##############################################################################
# Vlasov-Fluid solver

# Ion species
ion = Species(Tpar=Tpar, Tperp=Tperp)
species = (ion,)

# Electron fluid properties
electrons = Electrons(Te=Te)

# Initialize solver
vlasovfluid = VlasovFluid(B0, species, electrons)

solver1 = Solver(vlasovfluid)

##############################################################################
# Full Vlasov solver

# species
ions = Species(Tpar=Tpar, Tperp=Tperp)
electrons = Species(Tpar=Te, Tperp=Te, mass=5.4e-3, charge=-1)
species = (ions, electrons)

# Initialize solver
vlasov = Vlasov(B0, species)

solver2 = Solver(vlasov)

##############################################################################
# Call solvers

solver1(k0, 2.0, Nk, theta, guess)
solver1(k0, 1e-4, Nk, theta, guess)

solver2(k0, 2.0, Nk, theta, guess)
solver2(k0, 1e-4, Nk, theta, guess)


# Plotting
plt.figure(1)
plt.clf()
fig, axes = plt.subplots(ncols=2, num=1)

for m in range(2):
    kpar = np.cos(theta)*solver1.solutions[m]['k']
    omega = solver1.solutions[m]['omega']
    if m == 0:
        axes[1].plot(kpar, omega.real, 'C0-', label=r"Vlasov-Fluid")
    else:
        axes[1].plot(kpar, omega.real, 'C0-')
    axes[0].plot(kpar, omega.imag, 'C0-')
    axes[1].set_ylabel(r"$Re(\omega/\Omega)$")
    axes[0].set_ylabel(r"$Im(\omega/\Omega)$")

for m in range(2):
    kpar = np.cos(theta)*solver2.solutions[m]['k']
    omega = solver2.solutions[m]['omega']
    if m == 0:
        axes[1].plot(kpar, omega.real, 'C1--', label=r"Vlasov-Vlasov")
    else:
        axes[1].plot(kpar, omega.real, 'C1--')
    axes[0].plot(kpar, omega.imag, 'C1--')

axes[0].set_ylim(-0.05, 0.3)
axes[1].legend(frameon=False, loc='lower right')

for ax in axes:
    ax.set_xlim(0, 1.5)

plt.show()
