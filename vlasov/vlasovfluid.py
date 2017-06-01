from vlasov.cython.types import Complex
import numpy as np


class VlasovFluid():

    def __init__(self, B0, species, electrons):
        self.Bcross1 = np.zeros((3, 3))
        self.kk = np.zeros((3, 3))
        self.sigma = np.zeros((3, 3), dtype=Complex)
        self.D = np.zeros((3, 3), dtype=Complex)
        self.sigma_p = np.zeros((3, 3), dtype=Complex)
        self.D_p = np.zeros((3, 3), dtype=Complex)

        # Magnetic field strength
        self.B0 = B0

        # Initialize magnetic field matrix
        self.Bcross1[0, 1] = -self.B0
        self.Bcross1[1, 0] = +self.B0

        self.species = species

        self.electrons = electrons
        self.mu0 = 1

        # Mass density
        self.rho = 0

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
        for ion in self.species:
            b = np.array(lambda_tensor(omega, kpar, kperp, ion))
            self.sigma += 1j/omega*ion.charge**2*ion.n0/ion.mass * b

        self.sigma = np.reshape(self.sigma, (3, 3))

    def derivative_conductivity_tensor(self, omega, kpar, kperp):
        from vlasov.cython.lamb import derivative_lambda_tensor
        from vlasov.cython.lamb import lambda_tensor
        self.sigma_p[:] = 0.0
        self.sigma_p = np.reshape(self.sigma_p, 9)
        for ion in self.species:
            b = np.array(derivative_lambda_tensor(omega, kpar, kperp, ion))
            self.sigma_p += 1j/omega*ion.charge**2*ion.n0/ion.mass * b
        for ion in self.species:
            b = np.array(lambda_tensor(omega, kpar, kperp, ion))
            self.sigma_p += -1j/omega**2*ion.charge**2*ion.n0/ion.mass * b

        self.sigma_p = np.reshape(self.sigma_p, (3, 3))

    def dispersion_tensor(self, omega, kpar, kperp):
        self.get_kk(kpar, kperp)
        self.get_k2(kpar, kperp)
        self.conductivity_tensor(omega, kpar, kperp)
        self.D = np.dot(self.kk - self.k2 + 1j*self.mu0*omega*self.sigma,
                        self.Bcross1)
        self.D += np.dot(self.mu0*self.sigma, self.electrons.dpedne
                         / self.electrons.charge*self.kk)
        self.D -= 1j*omega*self.mu0*self.electrons.ene*np.identity(3)

    def derivative_dispersion_tensor(self, omega, kpar, kperp):
        self.get_k2(kpar, kperp)
        self.conductivity_tensor(omega, kpar, kperp)
        self.derivative_conductivity_tensor(omega, kpar, kperp)
        self.D_p = np.dot(1j*self.mu0*omega*self.sigma_p +
                          1j*self.mu0*self.sigma, self.Bcross1)
        self.D_p += np.dot(self.mu0*self.sigma_p, self.electrons.dpedne
                           / self.electrons.charge*self.kk)
        self.D_p -= 1j*self.mu0*self.electrons.ene*np.identity(3)

    def derivative_det(self, omega, kpar, kperp):
        self.derivative_dispersion_tensor(omega, kpar, kperp)
        self.dispersion_tensor(omega, kpar, kperp)

        f = -self.D[1, 2] * self.D[2, 1] * self.D_p[0, 0] + \
            self.D[1, 1] * self.D[2, 2] * self.D_p[0, 0] + \
            self.D[1, 2] * self.D[2, 0] * self.D_p[0, 1] - \
            self.D[1, 0] * self.D[2, 2] * self.D_p[0, 1] - \
            self.D[1, 1] * self.D[2, 0] * self.D_p[0, 2] + \
            self.D[1, 0] * self.D[2, 1] * self.D_p[0, 2] + \
            self.D[0, 2] * self.D[2, 1] * self.D_p[1, 0] - \
            self.D[0, 1] * self.D[2, 2] * self.D_p[1, 0] - \
            self.D[0, 2] * self.D[2, 0] * self.D_p[1, 1] + \
            self.D[0, 0] * self.D[2, 2] * self.D_p[1, 1] + \
            self.D[0, 1] * self.D[2, 0] * self.D_p[1, 2] - \
            self.D[0, 0] * self.D[2, 1] * self.D_p[1, 2] - \
            self.D[0, 2] * self.D[1, 1] * self.D_p[2, 0] + \
            self.D[0, 1] * self.D[1, 2] * self.D_p[2, 0] + \
            self.D[0, 2] * self.D[1, 0] * self.D_p[2, 1] - \
            self.D[0, 0] * self.D[1, 2] * self.D_p[2, 1] - \
            self.D[0, 1] * self.D[1, 0] * self.D_p[2, 2] + \
            self.D[0, 0] * self.D[1, 1] * self.D_p[2, 2]

        return f

    def det(self, omega, kpar, kperp):
        self.dispersion_tensor(omega, kpar, kperp)
        return np.linalg.det(self.D)
