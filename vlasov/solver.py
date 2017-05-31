import numpy as np
from .newton import newton
from vlasov.cython.types import Complex

class Solver:

    def __init__(self, disp, tol=1.48e-8, maxiter=100):
        self.disp = disp
        self.tol = tol
        self.maxiter = maxiter
        self.solutions = []

    def solve(self, guess, kpar, kperp):
        try:
            omega = newton(self.disp.det, guess, args=(kpar, kperp),
                           tol=self.tol, maxiter=self.maxiter)
        except:
            return None
        if np.abs(self.disp.det(omega, kpar, kperp)) < 1000*self.tol:
            return omega

    def __call__(self, kstart, kend, steps, theta, guess):
        k = np.linspace(kstart, kend, steps)
        kpar = k*np.cos(theta)
        kperp = k*np.sin(theta)


        omega = np.zeros(steps, dtype=Complex)
        omega[0] = guess
        omega[0] = self.solve(omega[0], kpar[0], kperp[0])

        for i in range(1, steps):
            result = self.solve(omega[i-1], kpar[i], kperp[i])
            if result is not None:
                    omega[i] = result
            else:
                break

        self.solutions.append({'omega': omega[:i-1], 'k': k[:i-1],
                               'theta': theta})

    def create_data(self):
        for m in range(self.M):
            k = self.solutions[m]['k']
            N = len(k)
            temp = np.empty((N, 3), dtype=Complex)
            temp[:, 0] = k
            temp[:, 1] = self.solutions[m]['theta']
            temp[:, 2] = self.solutions[m]['omega']
            if m == 0:
                data = temp
            else:
                data = np.concatenate((data, temp))
        return data

    @property
    def M(self):
        return len(self.solutions)

