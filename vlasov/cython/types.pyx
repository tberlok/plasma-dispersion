from types cimport real_t, complex_t
from numpy import dtype

Int = dtype("i{}".format(sizeof(int)))
Float = dtype("f{}".format(sizeof(real_t)))
Complex = dtype("c{}".format(sizeof(complex_t)))

cdef class species_t:
    pass
