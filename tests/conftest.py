import numpy
import pytest

from solvent_evaporation import Mixture, SoaveRedlichKwong


@pytest.fixture
def mixture(binary, srk_constants):
    R, T, P, v, _, _, _ = binary
    T_c, P_c, omega, l_AB, l_b_AB, V_bar = srk_constants
    excess_volume = SoaveRedlichKwong(R, T, P, v[0], T_c, P_c, omega,
                                      l=l_AB, l_b=l_b_AB)
    D0 = numpy.array([1.0, 3.0])                
    D_g = numpy.array([2.0, 4.0])               
    phi_g_sat = numpy.array([0.1, 0.005])       
    theta = numpy.array([1e-2, 2e-2])           
    return Mixture(phi_g_sat=phi_g_sat, alpha=1.0 / D_g, theta=theta, V_bar=V_bar,
                   D0=D0, excess_volume=excess_volume)


@pytest.fixture
def binary():
    R, T, P = 8.314462618, 300.0, 1.0e5
    v = numpy.array([5.0e-5, 2.0e-5])
    D0 = numpy.array([1.0, 2.0])
    V_bar = v / v[0]
    theta = v * P / (R * T)
    return R, T, P, v, D0, V_bar, theta


@pytest.fixture
def srk_constants():
    T_c = numpy.array([500.0, 500.0])
    P_c = numpy.array([50.0e5, 200.0e5])
    omega = numpy.array([0.5, 0.3])
    l_AB, l_b_AB = -0.05, 0.05
    V_bar = numpy.array([1.0, 2.0])
    return T_c, P_c, omega, l_AB, l_b_AB, V_bar


@pytest.fixture
def sine():
    n, delta_hat = 1000, 1.0
    intercept, coefficient = 0.5, 0.2
    eta = numpy.linspace(0.0, 1.0, n + 1)
    eta_face = (numpy.arange(n) + 0.5) / n
    phi = intercept + coefficient * numpy.sin(2 * numpy.pi * eta)
    phi_face_exact = intercept + coefficient * numpy.sin(2 * numpy.pi * eta_face)
    return n, delta_hat, intercept, coefficient, phi, eta_face, phi_face_exact
