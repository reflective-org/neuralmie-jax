"""Bulk optics of a lognormal population by Mie theory plus quadrature.

Transcribed from ``bulk_optics.py`` in pnnl/NEURALMIE @ 0b5b1d8 (BSD-2-Clause,
Copyright 2024 Battelle Memorial Institute), created by Andrew Geiss.

This defines exactly what the emulator was trained to predict::

    ke_rho = 0.75 * INT(Qe*pdf*r^2 dlnr) / INT(pdf*r^3 dlnr)      [1/m]
    ks_rho = 0.75 * INT(Qs*pdf*r^2 dlnr) / INT(pdf*r^3 dlnr)      [1/m]
    g      =        INT(g*Qs*pdf*r^2 dlnr) / INT(Qs*pdf*r^2 dlnr)

with ``pdf = exp(-(ln(r/mu))^2 / (2 ln^2(sigma_g)))`` -- the exponential part
of the lognormal, integrated in ``ln r`` so the missing ``1/r`` cancels
against the ``r`` of ``r dln r``. ``ke_rho`` is therefore the extinction
cross-section per unit *particle volume*; divide by the material density for
the mass extinction coefficient in m^2/kg.

``scipy`` is deliberately avoided: ``erfinv(2t-1)`` at the fixed quantiles is
obtained from the standard library's inverse normal CDF, which agrees to
~6e-16, so this package needs only numpy and jax.
"""

from __future__ import annotations

import statistics

import numpy as np

from neuralmie_jax.reference.tamie import coreshell, sphere

#: Radii in the numerical integration (``bulk_optics.py:8``).
NRAD = 1024

#: Fraction of the distribution captured by the bounds (``bulk_optics.py:9``).
INTFRAC = 0.999


def _erfinv_2t_minus_1(t: float) -> float:
    """``erfinv(2t-1)`` via the stdlib inverse normal CDF (scipy-free)."""
    return statistics.NormalDist().inv_cdf(t) / np.sqrt(2.0)


def integrate(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal integral of ``y`` over ``x`` (``bulk_optics.py:12-15``)."""
    dx = x[1:] - x[:-1]
    return float(np.sum(0.5 * dx * (y[1:] + y[:-1])))


def integration_bounds(mu: float, sigma_g: float, intfrac: float = INTFRAC) -> tuple[float, float]:
    """Radii bracketing ``intfrac`` of the lognormal (``bulk_optics.py:18-20``)."""
    t = np.array([(1.0 - intfrac) / 2.0, intfrac + (1.0 - intfrac) / 2.0])
    scale = np.array([_erfinv_2t_minus_1(float(ti)) for ti in t])
    lo, hi = np.exp(scale * np.log(sigma_g) * np.sqrt(2.0)) * mu
    return float(lo), float(hi)


def mass_efficiency(
    wavelength: float,
    mu: float,
    sigma_g: float,
    m_r: float,
    m_i: float,
    m_r_core: float | None = None,
    m_i_core: float | None = None,
    core_fraction: float | None = None,
    *,
    nrad: int = NRAD,
    intfrac: float = INTFRAC,
) -> tuple[float, float, float]:
    """Ground-truth bulk optics, in the upstream convention.

    Args:
        wavelength: wavelength in metres.
        mu: number-median radius in metres.
        sigma_g: geometric standard deviation.
        m_r: real refractive index (the shell, for a coated sphere).
        m_i: imaginary refractive index, positive.
        m_r_core: core real index; ``None`` selects the homogeneous solver.
        m_i_core: core imaginary index, positive.
        core_fraction: ``r_core/r_total``, which multiplies the size parameter.
        nrad: quadrature points; exposed only so convergence can be tested.
        intfrac: fraction of the distribution integrated over.

    Returns
    -------
        ``(ke_rho, ks_rho, g)`` with the first two in 1/m.
    """
    rmin, rmax = integration_bounds(mu, sigma_g, intfrac)
    r = np.exp(np.linspace(np.log(rmin), np.log(rmax), nrad))

    qe = np.zeros(nrad)
    qs = np.zeros(nrad)
    gg = np.zeros(nrad)
    for i in range(nrad):
        xs = 2.0 * np.pi * r[i] / wavelength
        if m_r_core is None:
            qe[i], qs[i], gg[i] = sphere(m_r + 1j * m_i, xs)
        else:
            qe[i], qs[i], gg[i] = coreshell(
                m_r_core + 1j * m_i_core, m_r + 1j * m_i, core_fraction * xs, xs
            )

    epdf = np.exp(-(np.log(r / mu) ** 2) / (2.0 * np.log(sigma_g) ** 2))
    lnr = np.log(r)
    vol = integrate(lnr, epdf * r**3)
    ke_rho = 0.75 * integrate(lnr, qe * epdf * r**2) / vol
    ks_rho = 0.75 * integrate(lnr, qs * epdf * r**2) / vol
    g = integrate(lnr, gg * qs * epdf * r**2) / integrate(lnr, qs * epdf * r**2)
    return ke_rho, ks_rho, g


def bulk_optics_reference(*args, **kwargs) -> tuple[float, float, float]:
    """As :func:`mass_efficiency`, re-expressed as ``(ke_rho, ssa, g)``."""
    ke_rho, ks_rho, g = mass_efficiency(*args, **kwargs)
    return ke_rho, ks_rho / ke_rho, g
