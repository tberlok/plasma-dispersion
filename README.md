# About this software
This is a dispersion solver for homogeneous plasmas with a gyrotropic
distribution function. The calculation of the  conductivity tensor for such
plasmas has been implemented in Cython in order to have optimal performance.

The Vlasov-Fluid (kinetic ions described by the collisionless Vlasov equation
and zero-mass electrons described as a fluid) dispersion relation has been
implemented as a class which can, given the frequency and the wavenumber,
return the determinant of the dispersion tensor.

This code is still work in progress but so far it can reproduce most of the
figures in the [HYDROS paper](https://arxiv.org/pdf/1605.07225.pdf),
"A linear dispersion relation for the hybrid
kinetic-ion/fluid-electron model of plasma physics" by
D. Told, J. Cookmeyer, P. Astfalk and F. Jenko. The code presented here does
not yet include resistivity but it is not limited to a single ion species.
I also hope to add support for drifting distribution functions in the near
future.

The code uses scipy's Newton-Raphson method which, however, falls back to the
secant method when the derivative of the function is not provided. I intend
to add the derivative of the lambda tensor which will hopefully aid in the
solver being a bit less sensitive to the initial guess for the frequency.

It should be easy to extend the present code to also have a class for a full
Vlasov model (that is, electrons also treated using the Vlasov equation),
as the calculation for the conductivity tensor is completely analogous.

Furthermore, it should also be simple to implement the standard textbook
plasma dispersion relation where the displacement current is included in the
derivation.

# How to use

Compilation can be done using make
```
$ make
```
and all the examples can be run by simply doing
```
$ make figures
```
