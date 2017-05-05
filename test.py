from vlasov import Species
from vlasov.cython.lamb import lambda_tensor
from vlasov.cython.types import Float, Complex
import numpy as np

ion = Species(Tpar=1, Tperp=1, B0=1)

a = np.zeros((3, 3), dtype=Complex)
omega = 2 + 1j
kpar = 1
kperp = 2.

lambda_tensor(omega, kpar, kperp, ion)
