from vlasov.cython.types import Complex
import numpy as np


class Vlasov():

    def __init__(self, B0, species, relativistic=False):
        self.kk = np.zeros((3, 3))
        self.sigma = np.zeros((3, 3), dtype=Complex)
        self.D = np.zeros((3, 3), dtype=Complex)

        # Magnetic field strength
        self.B0 = B0

        self.species = species

        self.mu0 = 1

        # Mass density
        self.rho = 0

        self.relativistic = relativistic
        # Speed of light
        self.c = 299792458.0

        for spec in species:
            # Cyclotron frequency
            spec.oc = spec.charge*self.B0/spec.mass
            self.rho += spec.mass*spec.n0

        # Alven velocity
        self.va = B0/np.sqrt(self.mu0*self.rho)

        # Calculate parallel and perpendicular beta
        for spec in species:
            spec.beta_para = 2*spec.vt**2/self.va**2
            spec.beta_perp = 2*spec.vtp**2/self.va**2

    def get_k2(self, kpar, kperp):
        self.k2 = np.identity(3)*(kpar**2 + kperp**2)

    def get_kk(self, kpar, kperp):
        self.kk[0, 0] = kperp**2
        self.kk[0, 2] = kpar*kperp
        self.kk[2, 0] = self.kk[0, 2]
        self.kk[2, 2] = kpar**2

    def conductivity_tensor(self, omega, kpar, kperp):
        from vlasov.cython.lamb import lambda_tensor
        self.sigma[:] = 0.0
        self.sigma = np.reshape(self.sigma, 9)
        for spec in self.species:
            b = np.array(lambda_tensor(omega, kpar, kperp, spec))
            self.sigma += 1j/omega*spec.charge**2*spec.n0/spec.mass * b

        self.sigma = np.reshape(self.sigma, (3, 3))

    def dispersion_tensor(self, omega, kpar, kperp):
        self.get_kk(kpar, kperp)
        self.get_k2(kpar, kperp)
        self.conductivity_tensor(omega, kpar, kperp)
        self.D = self.kk - self.k2 + 1j*self.mu0*omega*self.sigma
        if self.relativistic:
            self.D += (omega/self.c)**2*np.identity(3)

    def det(self, omega, kpar, kperp):
        self.dispersion_tensor(omega, kpar, kperp)
        return np.linalg.det(self.D)
