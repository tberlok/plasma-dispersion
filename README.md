# About this software
This is a dispersion solver for homogeneous plasmas with a gyrotropic
distribution function. The theory underlying the code is described in detail
in [my PhD thesis](https://ui.adsabs.harvard.edu/abs/2017PhDT.......356B/abstract).
See in particular Chapters 6, 7 and 12.

#### Some details

The calculation of the  conductivity tensor has been implemented in Cython
in order to have optimal performance.

The Vlasov-Fluid (kinetic ions described by the collisionless Vlasov equation
and zero-mass electrons described as a fluid) dispersion relation has been
implemented as a class which can, given the frequency and the wavenumber,
return the determinant of the dispersion tensor.

This code is still work in progress but so far it can reproduce most of the
figures in the [HYDROS paper](https://arxiv.org/pdf/1605.07225.pdf),
"A linear dispersion relation for the hybrid
kinetic-ion/fluid-electron model of plasma physics" by
D. Told, J. Cookmeyer, P. Astfalk and F. Jenko. Notable exceptions are the
plot for the oblique firehose instability and the fastwave which I still
need to figure out.

The code presented here does
not yet include resistivity but it is not limited to a single ion species.
At some point I also hope to add support for drifting distribution functions.

It might also prove useful to reproduce some of the plots in
the paper [Comparative study of gyrokinetic, hybrid-kinetic and
fully kinetic wave physics for
space plasmas](https://arxiv.org/pdf/1605.08385v1.pdf).

The code uses scipy's Newton-Raphson method which, however, falls back to the
secant method when the derivative of the function is not provided. I intend
to add the derivative of the lambda tensor which will hopefully aid in the
solver being a bit less sensitive to the initial guess for the frequency.

# Requirements
Assuming you already have Python 3 installed you can install the rest of the
requirements using:
```
$ pip install -r requirements.txt
```

# How to use
Compilation can be done using
```
$ make
```
It is also a good idea to the directory to your python path. On my computer
this would be
```
$ export PYTHONPATH=$PYTHONPATH:/Users/berlok/projects/plasma-dispersion
```
All the examples can be run by simply doing
```
$ make figures
```
