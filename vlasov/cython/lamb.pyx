# distutils: language = c++
cimport cython
from types cimport species_t
from types cimport real_t, complex_t
from libc.math cimport sqrt
from libc.math cimport M_PI as pi
from scipy.special.cython_special cimport ive
from scipy.special.cython_special cimport wofz

cdef extern from "<complex>":
    double abs(double complex)

cdef complex_t w_func(complex_t z):
    cdef complex_t  W = (1. + 1j*sqrt(0.5*pi)*z*wofz(sqrt(0.5)*z))
    return W


def lambda_tensor(complex_t omega, real_t kpar, real_t kperp, species_t s):

    cdef int N = 1000
    cdef real_t tol = 1e-16
    cdef int n, count, i, j, sign
    cdef complex_t T[9]
    cdef complex_t a[9]
    cdef complex_t zeta_0, zeta_n, square, Wn, temp
    cdef real_t Gam, GamP, lamb, lamb_sqrt

    lamb_sqrt = kperp*s.vtp/s.oc
    lamb = lamb_sqrt**2.0
    zeta_0 = omega/(kpar*s.vt)

    for i in range(9):
        a[i] = 0.0

    # This loop has n = 0, 1, -1, 2, -2, ... N, - N
    sign = -1
    for j in range(1, 2*(N+1)):
        n = j/2*sign
        sign *= -1
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

        count = 0
        for i in range(9):
            temp = square*T[i]
            if j > 1:
                if abs(temp/a[i]) > tol:
                    a[i] += temp
                    count += 1
            else:
                a[i] += temp
                count += 1
        if count == 0 and n < 0:
            break

    a[8] -= 1/(1-s.Delta)*zeta_0**2

    return a
