import numpy
import scipy

from solvent_evaporation import EvaporationModel, Grid, Margules, Mixture, SoaveRedlichKwong
   

def test_flux_ideal(binary, sine):
    
    _, _, P, _, D0, V_bar, theta = binary
    n, delta_hat, intercept, coefficient, phi, eta_face, phi_face_exact = sine

    def flux(eta):
        phi = intercept + coefficient * numpy.sin(2 * numpy.pi * eta)
        x_A = (phi / V_bar[0]) / (phi / V_bar[0] + (1 - phi) / V_bar[1])
        x_B = 1 - x_A
        D_AB = D0[1] ** x_A * D0[0] ** x_B
        return (-(D_AB / delta_hat) * 2 * numpy.pi * coefficient
                * numpy.cos(2 * numpy.pi * eta))

    mixture = Mixture(phi_g_sat=numpy.array([1.0e4, 5.0e3]) / P,
                      alpha=1.0 / numpy.array([2.0, 4.0]),
                      theta=theta, V_bar=V_bar, D0=D0,
                      activity=None, excess_volume=None)
    model = EvaporationModel(mixture, [0.5, 0.5], b_hat=2.0, liquid_grid=Grid(n=n),
                             gas_grid=Grid(n=n, start=1.0))

    j_A_exact = flux(eta_face)

    phi_face, _, j_A = model.liquid_faces(phi, delta_hat)
    assert numpy.allclose(phi_face, phi_face_exact)
    assert numpy.allclose(j_A, j_A_exact)

def test_flux_non_ideal(binary, srk_constants, sine):
    # Calculation of the non-ideal flux for a sine volume fraction field
    R, T, P, v, D0, V_bar, theta = binary
    T_c, P_c, omega, l_AB, l_b_AB, _ = srk_constants
    n, delta_hat, intercept, coefficient, phi, eta_face, phi_face_exact = sine
    RT = R * T
    M_A, M_B = 0.5, 0.3

    alpha_SRK = (1.0 + (0.480 + 1.574 * omega - 0.176 * omega ** 2)
                 * (1.0 - numpy.sqrt(T / T_c))) ** 2
    a = 0.42748 * alpha_SRK * (R * T_c) ** 2 / P_c
    b = 0.08664 * R * T_c / P_c
    a_AB = (1.0 - l_AB) * numpy.sqrt(a[0] * a[1])
    b_AB = (1.0 - l_b_AB) * (b[0] + b[1]) / 2
    a_ij = numpy.array([[a[0], a_AB], [a_AB, a[1]]])
    b_ij = numpy.array([[b[0], b_AB], [b_AB, b[1]]])

    def mixing(m, x_A):
        x_B = 1.0 - x_A
        return (x_A ** 2 * m[0, 0] + 2 * x_A * x_B * m[0, 1] + x_B ** 2 * m[1, 1],
                2 * (x_A * m[0, 0] + (x_B - x_A) * m[0, 1] - x_B * m[1, 1]),
                2 * (m[0, 0] - 2 * m[0, 1] + m[1, 1]))

    def root(x_A):
        a_tilde, beta = mixing(a_ij, x_A)[0] * P / RT ** 2, mixing(b_ij, x_A)[0] * P / RT
        linear = a_tilde - beta - beta ** 2

        def cubic(Z):
            return Z ** 3 - Z ** 2 + linear * Z - a_tilde * beta
        peak = (1.0 - numpy.sqrt(1.0 - 3.0 * linear)) / 3.0
        return scipy.optimize.brentq(cubic, beta, peak)

    def slopes(x_A):
        Z = root(x_A)
        a_t, a_t_x, a_t_xx = (value * P / RT ** 2 for value in mixing(a_ij, x_A))
        beta, beta_x, beta_xx = (value * P / RT for value in mixing(b_ij, x_A))
        C_Z = 3 * Z ** 2 - 2 * Z + a_t - beta - beta ** 2
        Z_x = (((1 + 2 * beta) * Z + a_t) * beta_x - (Z - beta) * a_t_x) / C_Z
        Z_xx = -((6 * Z - 2) * Z_x ** 2 + 2 * (a_t_x - (1 + 2 * beta) * beta_x) * Z_x
                 - 2 * beta_x * (a_t_x + Z * beta_x) + (Z - beta) * a_t_xx
                 - ((1 + 2 * beta) * Z + a_t) * beta_xx) / C_Z
        return Z, Z_x, Z_xx

    def worked(x_A):
        Z_A, Z_B = root(1.0), root(0.0)
        Z, Z_x, Z_xx = slopes(x_A)
        x_B = 1.0 - x_A
        return (Z - x_A * (Z_A - theta[0]) - x_B * (Z_B - theta[1]),
                Z_x - (Z_A - theta[0]) + (Z_B - theta[1]), Z_xx)

    def phi_x(x_A):
        Z, Z_x, _ = worked(x_A)
        return x_A * (Z + (1 - x_A) * Z_x) / Z

    def flux(eta):
        phi = intercept + coefficient * numpy.sin(2 * numpy.pi * eta)
        x_A = scipy.optimize.brentq(lambda x: phi_x(x) - phi, 0.0, 1.0)
        x_B = 1 - x_A
        Z, Z_x, Z_xx = worked(x_A)
        curvature = x_A * x_B * Z * Z_xx
        s = curvature / ((Z + x_B * Z_x) * (Z - x_A * Z_x) + curvature)
        Gamma = 1 - 2 * x_A * x_B * (M_A * (2 * x_B - x_A) + M_B * (2 * x_A - x_B))
        D_AB = D0[1] ** x_A * D0[0] ** x_B * Gamma
        return (-(D_AB / delta_hat) * (1 - s) * 2 * numpy.pi * coefficient
                * numpy.cos(2 * numpy.pi * eta))
    
    j_A_exact = numpy.array([flux(e) for e in eta_face])
    
    # Code generated result for the non-ideal flux using the sine volume fraction field
    srk = SoaveRedlichKwong(R, T, P, v[0], T_c, P_c, omega, l=l_AB, l_b=l_b_AB)
    mixture = Mixture(phi_g_sat=numpy.array([1.0e4, 5.0e3]) / P,
                      alpha=1.0 / numpy.array([2.0, 4.0]),
                      theta=theta, V_bar=V_bar, D0=D0,
                      activity=Margules(M_A, M_B), excess_volume=srk)
    model = EvaporationModel(mixture, [0.5, 0.5], b_hat=2.0, liquid_grid=Grid(n=n),
                             gas_grid=Grid(n=n, start=1.0)) 
    
    phi_face, _, j_A = model.liquid_faces(phi, delta_hat)
    assert numpy.allclose(phi_face, phi_face_exact)
    assert numpy.allclose(j_A, j_A_exact)