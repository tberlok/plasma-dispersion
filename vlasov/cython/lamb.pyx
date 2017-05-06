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

cdef extern from "<complex>":
    double abs(double complex)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lambda_tensor(complex_t omega, real_t kpar, real_t kperp, species_t s):

    cdef int maxterms = 1000
    cdef int n_arr[2003]
    cdef real_t Gamma[2003]
    cdef real_t GammaP[2001]
    cdef real_t tol = 1e-12
    cdef int n, count, i, j
    cdef complex_t T[9]
    cdef complex_t a[9]
    cdef complex_t zeta_0, zeta_n, square, Wn, temp
    cdef real_t Gam, GamP, lamb, lamb_sqrt

    lamb_sqrt = kperp*s.vtp/s.oc
    lamb = lamb_sqrt**2.0
    zeta_0 = omega/(kpar*s.vt)

    for i in range(9):
        a[i] = 0.0

    n_arr[0] = 0
    for j in range(1, maxterms+2):
        n_arr[2*j] = j
        n_arr[2*j-1] = -j

    for j in range(2003):
        Gamma[j] = ive(n_arr[j], lamb)

    GammaP[0] = 0.5*(Gamma[2] + Gamma[1]) - Gamma[0]
    GammaP[1] = 0.5*(Gamma[0] + Gamma[3]) - Gamma[1]
    for j in range(2, 2001):
        GammaP[j] = 0.5*(Gamma[j-2] + Gamma[j+2]) - Gamma[j]

    # for j in range(2001):
    #     n = n_arr[j]
    #     GamP = 0.5*(ive(n-1, lamb) + ive(n+1, lamb)) - ive(n, lamb)
    #     print(j, n, GamP, GammaP[j])

    for j in range(2001):
        n = n_arr[j]
        zeta_n = (omega - n*s.oc)/(kpar*s.vt)
        Wn = w_func(zeta_n)
        Gam = Gamma[j]
        GamP = GammaP[j]

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

        count = 0
        for i in range(9):
            temp = square*T[i]
            if j > 0:
                if abs(temp/a[i]) > tol:
                    a[i] += temp
                    count += 1
            else:
                a[i] += temp
                count += 1
        if count == 0:
            break

    a[8] -= 1/(1-s.Delta)*zeta_0**2

    return a
