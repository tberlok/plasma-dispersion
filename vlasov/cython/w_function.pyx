# distutils: language = c++
# distutils: sources = vlasov/cython/Faddeeva.cc

# Plasma dispersion function W (zeta)
from libc.math cimport sqrt
from libc.math cimport M_PI as pi
from scipy.special.cython_special import ive

cdef extern from "Faddeeva.hh" namespace "Faddeeva":
    double complex w(double complex)

cpdef double complex w_func(double complex z):
    cdef double complex  W = (1. + 1j*sqrt(0.5*pi)*z*w(sqrt(0.5)*z))
    return W

cpdef Gamma(double n, double x):
    return ive(n, x)

cpdef GammaP(double n, double lamb):
    return 0.5*(ive(n-1, lamb) + ive(n+1, lamb)) - ive(n, lamb)
