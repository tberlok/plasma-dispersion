ctypedef double real_t
ctypedef double complex complex_t


cdef class species_t:
    cdef public real_t mass, charge, n, v0, Tpar, Tperp
    cdef public real_t Delta, vt, vtp, oc