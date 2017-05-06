from .cython.types import species_t
import numpy as np


class Species(species_t):

    def __init__(self, Tpar, Tperp, mass=1, charge=1, n0=1, v0=0):

        self.mass = mass
        self.charge = charge
        # Number density
        self.n0 = n0
        # Drift velocity
        self.v0 = v0
        # Parallel and perpendicular pressure
        self.Tpar = Tpar
        self.Tperp = Tperp

        # Temperature anisotropy parameter
        self.Delta = 1 - Tperp/Tpar
        # Parallel thermal velocity
        self.vt = np.sqrt(Tpar/mass)
        self.vtp = np.sqrt(Tperp/mass)
