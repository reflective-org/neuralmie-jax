"""TAMie: Mie scattering for homogeneous and coated spheres (NumPy reference).

Transcribed verbatim from ``TAMie.py`` in pnnl/NEURALMIE @ 0b5b1d8
(BSD-2-Clause, Copyright 2024 Battelle Memorial Institute), created by
Andrew Geiss, 07 Feb 2023. Coated-sphere algorithm: Toon, O. B. and
Ackerman, T. P., Appl. Opt. 20, 3657-3660, 1981.

This is the **ground truth** the emulator is validated against, and it is
deliberately NumPy rather than JAX: the recurrences are sequential, need
complex float64, and use a data-dependent order
``nmax = int(m.real*x + 4.3*(m.real*x)**(1/3) + 3)`` that cannot be traced.

Every expression is transcribed as-is. Several look like they could be
tidied and must not be:

* ``dlnpsi`` allocates ``nmax+20`` and starts the downward recurrence at
  ``nmax+18``. That lead-in is the downward-stabilisation margin.
* ``nmax`` uses ``m.real*x``, not ``x`` (Wiscombe's criterion uses ``x``).
* ``qqg`` drops the last order via ``[:-1]`` and takes
  ``n = arange(1, len(a))``: a deliberate off-by-one that pairs ``a[n]``
  with ``a[n+1]`` for the asymmetry sum.
* Both solvers force ``m = m.real - 1j*abs(m.imag)`` internally, while
  callers pass a *positive* imaginary index.
* ``coreshell`` short-circuits on ``mc == ms`` (exact complex equality),
  ``xc/xs < 0.01`` and ``xc/xs > 0.99``; the last returns the *core* index
  with the *shell* size, which is intentional. Note the emulator's domain
  caps the core fraction at 0.98, just under that trip point.

numba is optional and only accelerates it; results are identical without.
"""

from __future__ import annotations

import numpy as np
from numpy import conj, exp, zeros

try:  # pragma: no cover - exercised by whichever branch the env provides
    from numba import njit

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """No-op stand-in supporting both ``@njit`` and ``@njit(cache=True)``."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

    
#downward recurrence for the logarithmic derivative of psi (psi'/psi):
@njit(cache=True)
def dlnpsi(z,nmax):
    eta = zeros((nmax+20),dtype='cdouble')
    for n in range(nmax+20-2,0,-1):
        eta[n] = (n+1)/z - 1/((n+1)/z + eta[n+1])
    return eta[:nmax]

#upward recurrence for the logarithmic derivative of zeta (zeta'/zeta):
@njit(cache=True)
def dlnzeta(z,nmax):
    eta = zeros((nmax),dtype='cdouble')
    eta[0] = -1j
    for n in range(1,nmax):
        eta[n] = -n/z+1/(n/z-eta[n-1])
    return eta

#upward recurrence for the ratio (T=psi/zeta)
@njit(cache=True)
def tup(z,dp,dz):
    T = zeros((len(dp)),dtype='cdouble')
    T[1] = 0.5*exp(2*1j*z)*(z+1j)/(z-1j) + 0.5
    for n in range(2,len(dp)):
        T[n] = T[n-1]*(n/z - dp[n-1])/(n/z - dz[n-1])
    return T

#calculate extinction and scattering efficiencies and asymmetry parameter from Mie coefficients
@njit(cache=True)
def qqg(a,b,x):
    n = np.arange(1,len(a))
    qe = 2*np.sum((2*n+1)*(a+b).real[:-1])/(x**2)
    qs = 2*np.sum((2*n+1)*(a.real**2+a.imag**2+b.real**2+b.imag**2)[:-1])/(x**2)
    g = 4*np.sum(((n+2)/(1+1/n)) * (a[:-1]*conj(a[1:]) + b[:-1]*conj(b[1:])).real + 
                 ((2+1/n)/(n+1)) * (a*conj(b)).real[:-1])/(qs*x**2)
    return qe, qs, g

#Mie calculations for a sphere
@njit(cache=True)
def sphere(m,x):
    m = m.real - 1j*np.abs(m.imag)
    nmax = int(m.real*x + 4.3*(m.real*x)**(1/3) + 3)
    dpx = dlnpsi(x,nmax)
    dpmx = dlnpsi(m*x,nmax)
    dzx = dlnzeta(x,nmax)
    T = tup(x,dpx,dzx)
    an = (T*(m*dpx-dpmx)/(m*dzx-dpmx))[1:]
    bn = (T*(dpx-m*dpmx)/(dzx-m*dpmx))[1:]
    return qqg(an,bn,x)

#Toon and Ackerman's algorithm for coated spheres
@njit(cache=True)
def coreshell(mc,ms,xc,xs):
    
    #check if the sphere function can be used instead
    if mc == ms or xc/xs < 0.01:
        return sphere(ms,xs)
    if xc/xs > 0.99:
        return sphere(mc,xs)
    
    #ensure index of refraction uses the correct sign convention
    mc = mc.real - 1j*np.abs(mc.imag)
    ms = ms.real - 1j*np.abs(ms.imag)
    
    #logarithmic derivatives psi'/psi and zeta'/zeta by recurrence:
    #naming convention: d = log derivative, Z = zeta, P = psi, sc = m_s*x_c etc., 's' alone = x_s
    nmax = int(ms.real*xs + 4.3*(ms.real*xs)**(1/3) + 3)
    [dPss,dPs,dPcc,dPsc] =  [dlnpsi(arg,nmax) for arg in [ms*xs, xs+1j*0, mc*xc,ms*xc]]
    [dZss,dZs,     dZsc] = [dlnzeta(arg,nmax) for arg in [ms*xs, xs+1j*0,       ms*xc]]
        
    #upward recurrence for zeta(msxc)*psi(msxc), zeta(msxs)*psi(msxc), psi(msxc)/psi(msxs), and psi(xs)/zeta(xs):
    PsoZs = tup(xs,dPs,dZs)
    ZscPsc, ZssPsc, PscoPss = [zeros((nmax),dtype='cdouble') for _ in range(3)]
    ZscPsc[0]  = (1-exp(-2*1j*ms*xc))/2
    ZssPsc[0]  = -0.5*(exp(-1j*ms*(xs+xc)) - exp(-1j*ms*(xs-xc)))
    PscoPss[0] = (exp(-1j*ms*(xc+xs)) - exp(1j*ms*(xc-xs)))/(exp(-2*1j*ms*xs)-1)
    for n in range(1,nmax):
        PscoPss[n] = PscoPss[n-1]*(dPss[n]+n/(ms*xs))/(dPsc[n]+n/(ms*xc))
        ZscPsc[n]  = ZscPsc[n-1]/((dZsc[n] + n/(ms*xc))*(dPsc[n]+n/(ms*xc)))
        ZssPsc[n]  = ZssPsc[n-1]/((dZss[n] + n/(ms*xs))*(dPsc[n]+n/(ms*xc)))
    
    #mie coefficients:
    U = mc*dPsc - ms*dPcc
    V = ms*dPsc - mc*dPcc
    W = -1j*(PscoPss*ZssPsc - ZscPsc)
    an = (PsoZs*((dPss-ms*dPs)*(mc+U*W) - U*PscoPss**2 ) / ((dPss-ms*dZs)*(mc+U*W) - U*PscoPss**2 ))[1:]
    bn = (PsoZs*((ms*dPss-dPs)*(ms+V*W)-ms*V*PscoPss**2) / ((ms*dPss-dZs)*(ms+V*W)-ms*V*PscoPss**2))[1:]

    return qqg(an,bn,xs)