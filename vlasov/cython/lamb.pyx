# distutils: language = c++
# distutils: sources = vlasov/cython/Faddeeva.cc
cimport cython
from types cimport species_t
from types cimport real_t, complex_t
from libc.math cimport sqrt
from libc.math cimport M_PI as pi
from scipy.special.cython_special cimport ive

cdef extern from "Faddeeva.hh" namespace "Faddeeva":
    double complex w(double complex)

cdef double complex w_func(double complex z):
    cdef double complex  W = (1. + 1j*sqrt(0.5*pi)*z*w(sqrt(0.5)*z))
    return W

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lambda_tensor(complex_t omega, real_t kpar, real_t kperp, species_t s):

    cdef int maxterms = 1000
    cdef int n
    cdef complex_t T[9], a[9]
    cdef complex_t zeta_0, zeta_n, square, Wn
    cdef real_t Gam, GamP, lamb, lamb_sqrt

    lamb_sqrt = kperp*s.vtp/s.oc
    lamb = lamb_sqrt**2.0
    zeta_0 = omega/(kpar*s.vt)

    for j in range(9):
        a[j] = 0.0

    for n in range(-maxterms, maxterms + 1):
        zeta_n = (omega - n*s.oc)/(kpar*s.vt)
        Wn = w_func(zeta_n)
        Gam = ive(n, lamb)
        GamP = 0.5*(ive(n-1, lamb) + ive(n+1, lamb)) - ive(n, lamb)

        square = omega/(omega-n*s.oc)*(1.0-Wn) + s.Delta*Wn
        T[0] = n**2*Gam/lamb
        T[1] = 1j*n*GamP
        T[2] = n*Gam/sqrt(lamb)*zeta_n/sqrt(1-s.Delta)
        T[3] = T[1].conjugate()
        T[4] = n**2*Gam/lamb - 2*GamP*lamb
        T[5] = -1j*sqrt(lamb)*GamP*zeta_n/sqrt(1-s.Delta)
        T[6] = T[2].conjugate()
        T[7] = T[5].conjugate()
        T[8] = Gam*zeta_n**2/(1-s.Delta)

        for j in range(9):
            a[j] += square*T[j]

    a[8] -= 1/(1-s.Delta)*zeta_0**2

    return a
