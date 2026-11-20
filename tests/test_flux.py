import numpy
import scipy

from solvent_evaporation import EvaporationModel, Grid, Margules, Mixture, SoaveRedlichKwong
   

def test_flux_ideal(binary, sine):
    
    _, _, P, _, D0, v_i, theta = binary
    n, delta_hat, intercept, coefficient, phi, eta_face, phi_face_exact = sine

    def flux(eta):
        phi = intercept + coefficient * numpy.sin(2 * numpy.pi * eta)
        x_A = (phi / v_i[0]) / (phi / v_i[0] + (1 - phi) / v_i[1])
        x_B = 1 - x_A
        D_AB = D0[1] ** x_A * D0[0] ** x_B
        return (-(D_AB / delta_hat) * 2 * numpy.pi * coefficient
                * numpy.cos(2 * numpy.pi * eta))

    mixture = Mixture(phi_g_sat=numpy.array([1.0e4, 5.0e3]) / P,
                      alpha=1.0 / numpy.array([2.0, 4.0]),
                      theta=theta, v_i=v_i, D0=D0,
                      activity=None, excess_volume=None)
    model = EvaporationModel(mixture, [0.5, 0.5], b_hat=2.0, liquid_grid=Grid(n=n),
                             gas_grid=Grid(n=n, start=1.0))

    j_A_exact = flux(eta_face)

    phi_face, _, j_A = model.liquid_faces(phi, delta_hat)
    assert numpy.allclose(phi_face, phi_face_exact)
    assert numpy.allclose(j_A, j_A_exact)

def test_flux_non_ideal(binary, srk_constants, sine):
    # Calculation of the non-ideal flux for a sine volume fraction field
    R, T, P, v, D0, v_i, theta = binary
    T_c, P_c, omega, l_AB, l_b_AB, _ = srk_constants
    n, delta_hat, intercept, coefficient, phi, eta_face, phi_face_exact = sine
    RT = R * T
    A_12, A_21 = 0.5, 0.3

    alpha_SRK = (1.0 + (0.480 + 1.574 * omega - 0.176 * omega ** 2)
                 * (1.0 - numpy.sqrt(T / T_c))) ** 2
    a_i = 0.42748 * alpha_SRK * (R * T_c) ** 2 / P_c
    b_i = 0.08664 * R * T_c / P_c
    a_AB = (1.0 - l_AB) * numpy.sqrt(a_i[0] * a_i[1])
    b_AB = (1.0 - l_b_AB) * (b_i[0] + b_i[1]) / 2
    a_ij = numpy.array([[a_i[0], a_AB], [a_AB, a_i[1]]])
    b_ij = numpy.array([[b_i[0], b_AB], [b_AB, b_i[1]]])

    def mixing(m, x_A):
        x_B = 1.0 - x_A
        return (x_A ** 2 * m[0, 0] + 2 * x_A * x_B * m[0, 1] + x_B ** 2 * m[1, 1],
                2 * (x_A * m[0, 0] + (x_B - x_A) * m[0, 1] - x_B * m[1, 1]),
                2 * (m[0, 0] - 2 * m[0, 1] + m[1, 1]))

    def root(x_A):
        a, b = mixing(a_ij, x_A)[0], mixing(b_ij, x_A)[0]
        linear = a - RT * b - P * b ** 2

        def cubic(v):
            return P * v ** 3 - RT * v ** 2 + linear * v - a * b
        peak = (RT - numpy.sqrt(RT ** 2 - 3.0 * P * linear)) / (3.0 * P)
        return scipy.optimize.brentq(cubic, b, peak)

    def slopes(x_A):
        v = root(x_A)
        a, a_x, a_xx = mixing(a_ij, x_A)
        b, b_x, b_xx = mixing(b_ij, x_A)
        C_v = 3 * P * v ** 2 - 2 * RT * v + a - RT * b - P * b ** 2
        v_x = (((RT + 2 * P * b) * v + a) * b_x - (v - b) * a_x) / C_v
        v_xx = -((6 * P * v - 2 * RT) * v_x ** 2 + 2 * (a_x - (RT + 2 * P * b) * b_x) * v_x
                 - 2 * b_x * (a_x + P * v * b_x) + (v - b) * a_xx
                 - ((RT + 2 * P * b) * v + a) * b_xx) / C_v
        return v, v_x, v_xx

    def phi_x(x_A):
        v, v_x, _ = slopes(x_A)
        return x_A * (v + (1 - x_A) * v_x) / v

    def flux(eta):
        phi = intercept + coefficient * numpy.sin(2 * numpy.pi * eta)
        x_A = scipy.optimize.brentq(lambda x: phi_x(x) - phi, 0.0, 1.0)
        x_B = 1 - x_A
        v, v_x, v_xx = slopes(x_A)
        curvature = x_A * x_B * v * v_xx
        s = curvature / ((v + x_B * v_x) * (v - x_A * v_x) + curvature)
        Gamma = 1 - 2 * x_A * x_B * (A_12 * (2 * x_B - x_A) + A_21 * (2 * x_A - x_B))
        D_AB = D0[1] ** x_A * D0[0] ** x_B * Gamma
        return (-(D_AB / delta_hat) * (1 - s) * 2 * numpy.pi * coefficient
                * numpy.cos(2 * numpy.pi * eta))
    
    j_A_exact = numpy.array([flux(e) for e in eta_face])
    
    # Code generated result for the non-ideal flux using the sine volume fraction field
    srk = SoaveRedlichKwong(R, T, P, v[0], T_c, P_c, omega, l=l_AB, l_b=l_b_AB)
    mixture = Mixture(phi_g_sat=numpy.array([1.0e4, 5.0e3]) / P,
                      alpha=1.0 / numpy.array([2.0, 4.0]),
                      theta=theta, v_i=v_i, D0=D0,
                      activity=Margules(A_12, A_21), excess_volume=srk)
    model = EvaporationModel(mixture, [0.5, 0.5], b_hat=2.0, liquid_grid=Grid(n=n),
                             gas_grid=Grid(n=n, start=1.0)) 
    
    phi_face, _, j_A = model.liquid_faces(phi, delta_hat)
    assert numpy.allclose(phi_face, phi_face_exact)
    assert numpy.allclose(j_A, j_A_exact)