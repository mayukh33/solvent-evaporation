"""The made-up binary of the tests, with round numbers and every quantity over
its scale, but for its excess volume: the Soave-Redlich-Kwong one of ethanol
and water at 298.15 K and 1 atm, over v_ethanol, with ethanol_water's
constants."""

import numpy
import pytest

from solvent_evaporation import Mixture, SoaveRedlichKwong


@pytest.fixture
def mixture(ethanol_water):
    R, T, P, v, _, _, _ = ethanol_water
    T_c = numpy.array([513.9, 647.1])                # K, SVA Table B.1
    P_c = numpy.array([61.48e5, 220.55e5])           # Pa
    omega = numpy.array([0.645, 0.345])
    excess_volume = SoaveRedlichKwong(R, T, P, v[0], T_c, P_c, omega,
                                      l=-0.07865, l_b=0.03972)
    V_bar = numpy.array([1.0, 2.0])             # molar volumes, B's twice A's
    D0 = numpy.array([1.0, 3.0])                # A dilute in B, B dilute in A
    D_g = numpy.array([2.0, 4.0])               # each vapour in air
    phi_g_sat = numpy.array([0.1, 0.005])       # the interface's only
    theta = numpy.array([1e-2, 2e-2])           # the interface's only
    return Mixture(phi_g_sat=phi_g_sat, alpha=1.0 / D_g, theta=theta, V_bar=V_bar,
                   D0=D0, excess_volume=excess_volume)


@pytest.fixture
def ethanol_water():
    R, T, P = 8.314462618, 298.15, 101325.0
    v = numpy.array([46.068e-3 / 784.95, 18.015e-3 / 997.05])
    D0 = numpy.array([1.24e-9, 1.13e-9]) / 1.24e-9
    V_bar = v / v[0]
    theta = v * P / (R * T)
    return R, T, P, v, D0, V_bar, theta


@pytest.fixture
def sine():
    n, delta_hat = 1000, 1.0
    intercept, coefficient = 0.5, 0.2
    eta = numpy.linspace(0.0, 1.0, n + 1)
    eta_face = (numpy.arange(n) + 0.5) / n
    phi = intercept + coefficient * numpy.sin(2 * numpy.pi * eta)
    phi_face_exact = intercept + coefficient * numpy.sin(2 * numpy.pi * eta_face)
    return n, delta_hat, intercept, coefficient, phi, eta_face, phi_face_exact
