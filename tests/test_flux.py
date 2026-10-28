import numpy
import pytest


n = 500                             # cells in each phase
V_A, V_B = 1.0, 2.0                 # molar volumes, B's twice A's
D0_AB, D0_BA = 1.0, 3.0             # A dilute in B, B dilute in A
D_g = numpy.array([2.0, 4.0])       # each vapour in air
theta = numpy.array([1e-2, 2e-2])   # pure liquid over gas molar volume
b_hat = 2.0                         # outer edge of the gas


@pytest.fixture
def liquid_mesh():
    """Centres and faces of the liquid cells, by hand from their count."""
    return (numpy.arange(n) + 0.5) / n, numpy.arange(n + 1) / n


@pytest.fixture
def gas_mesh():
    """Centres and faces of the gas cells, eta 1 to 2, by hand."""
    return 1.0 + (numpy.arange(n) + 0.5) / n, 1.0 + numpy.arange(n + 1) / n


def liquid(phi_A):
    """phi, (2, ...), from phi_A: B makes up the rest."""
    return numpy.array([phi_A, 1.0 - phi_A])


def diffusivity(phi_A):
    """D_AB at volume fraction phi_A, written out by hand.

    D_AB = D_MS, the mixture being ideal (Gamma = 1): Vignes gives the
    Maxwell-Stefan D_MS = (D0_AB)^x_B (D0_BA)^x_A.  The partial molar volumes
    being constant, the volume-frame flux is exactly j_A = -D_AB dphi_A/dy.
    """
    C_A, C_B = phi_A / V_A, (1.0 - phi_A) / V_B
    x_A = C_A / (C_A + C_B)
    x_B = 1.0 - x_A
    return D0_AB ** x_B * D0_BA ** x_A


def test_diffusivities(ideal):
    # phi_A = 0 and 1: D_AB = D0_AB and D0_BA
    pure_B = ideal.fick_matrix(liquid(0.0))[0, 0]
    pure_A = ideal.fick_matrix(liquid(1.0))[0, 0]
    assert numpy.isclose(pure_B, D0_AB)
    assert numpy.isclose(pure_A, D0_BA)

    # x_A = 1/2: D_AB = sqrt(D0_AB D0_BA)
    phi_A = V_A / (V_A + V_B)
    D_MS = numpy.sqrt(D0_AB * D0_BA)
    assert numpy.isclose(ideal.fick_matrix(liquid(phi_A))[0, 0], D_MS)


def test_liquid_ideal(model, delta_hat, liquid_mesh):
    eta_l, faces_l = liquid_mesh
    no_flux = numpy.zeros(2)

    # phi_A = 0, 1 and 0.4, uniform: N_l = 0
    for phi_A in numpy.array([0.0, 1.0, 0.4]):
        N_l = model.liquid_faces(liquid(numpy.full_like(eta_l, phi_A)),
                                 delta_hat, no_flux, 0.0)
        assert numpy.allclose(N_l, 0.0)

    # phi_A = 0.2 + 0.6 eta: N_l = -D_AB(phi_A) 0.6/delta_hat
    N_l = model.liquid_faces(liquid(0.2 + 0.6 * eta_l), delta_hat, no_flux,
                             0.0)[0]
    by_hand = -diffusivity(0.2 + 0.6 * faces_l) * 0.6 / delta_hat
    assert numpy.allclose(N_l[1:-1], by_hand[1:-1])

    # phi_A = 0.5 + sin(2 pi eta)/4: exact face average and centred difference
    phi_A = 0.5 + 0.25 * numpy.sin(2.0 * numpy.pi * eta_l)
    N_l = model.liquid_faces(liquid(phi_A), delta_hat, no_flux, 0.0)[0]
    h = 1.0 / numpy.size(eta_l)
    phi_A_bar = (0.5 + 0.25 * numpy.sin(2.0 * numpy.pi * faces_l)
                 * numpy.cos(numpy.pi * h))
    dphi_A_dy = (0.5 * numpy.sin(numpy.pi * h)
                 * numpy.cos(2.0 * numpy.pi * faces_l) / (h * delta_hat))
    by_hand = -diffusivity(phi_A_bar) * dphi_A_dy
    assert numpy.allclose(N_l[1:-1], by_hand[1:-1])


def test_gas(model, delta_hat, gas_mesh):
    eta_g, faces_g = gas_mesh
    L_hat = b_hat - delta_hat
    no_flux = numpy.zeros(2)

    # phi_g = 0 against phi_g_inf = 0: N_g = 0
    N_g = model.gas_faces(numpy.zeros((2, numpy.size(eta_g))), L_hat,
                          no_flux, 0.0, 0.0)
    assert numpy.allclose(N_g, 0.0)

    # phi_g = phi_g_inf, uniform: N_g = 0
    phi_g_inf = numpy.array([0.3, 0.1])
    model.phi_g_ambient = phi_g_inf
    N_g = model.gas_faces(numpy.outer(phi_g_inf, numpy.ones_like(eta_g)), L_hat,
                          no_flux, 0.0, 0.0)
    assert numpy.allclose(N_g, 0.0)

    # phi_g = phi_g_s (2-eta) + phi_g_inf (eta-1):
    # N_g = D_g (phi_g_s - phi_g_inf)/L_hat
    phi_g_s, phi_g_inf = numpy.array([0.5, 0.2]), numpy.array([0.1, 0.0])
    model.phi_g_ambient = phi_g_inf
    phi_g = (numpy.outer(phi_g_s, 2.0 - eta_g)
             + numpy.outer(phi_g_inf, eta_g - 1.0))
    N_g = model.gas_faces(phi_g, L_hat, no_flux, 0.0, 0.0)
    by_hand = numpy.outer(-D_g * (phi_g_inf - phi_g_s) / L_hat,
                          numpy.ones_like(eta_g))
    assert numpy.allclose(N_g[:, 1:], by_hand)

    # phi_g = 0.5 + sin(2 pi eta)/4: exact centred difference across each face
    phi_g = numpy.outer(numpy.ones(2),
                        0.5 + 0.25 * numpy.sin(2.0 * numpy.pi * eta_g))
    N_g = model.gas_faces(phi_g, L_hat, no_flux, 0.0, 0.0)
    h = 1.0 / numpy.size(eta_g)
    dphi_g_dy = (0.5 * numpy.sin(numpy.pi * h)
                 * numpy.cos(2.0 * numpy.pi * faces_g) / (h * L_hat))
    by_hand = -numpy.outer(D_g, dphi_g_dy)
    assert numpy.allclose(N_g[:, 1:-1], by_hand[:, 1:-1])


def test_moving_surface(model, delta_hat, liquid_mesh, gas_mesh):
    eta_l, faces_l = liquid_mesh
    eta_g, faces_g = gas_mesh
    L_hat = b_hat - delta_hat
    no_flux = numpy.zeros(2)
    ddelta_dt, v_star_g = -0.3, 1.2     # the film thinning; the gas's velocity

    # phi_A = 0.4, uniform: N_l = -eta delta' phi_A
    N_l = model.liquid_faces(liquid(numpy.full_like(eta_l, 0.4)), delta_hat,
                             no_flux, ddelta_dt)[0]
    by_hand = -faces_l * ddelta_dt * 0.4
    assert numpy.allclose(N_l[1:-1], by_hand[1:-1])

    # phi_A = 0.2 + 0.6 eta: N_l = -D_AB 0.6/delta_hat - eta delta' phi_A
    phi_A_f = 0.2 + 0.6 * faces_l
    N_l = model.liquid_faces(liquid(0.2 + 0.6 * eta_l), delta_hat, no_flux,
                             ddelta_dt)[0]
    by_hand = (-diffusivity(phi_A_f) * 0.6 / delta_hat
               - faces_l * ddelta_dt * phi_A_f)
    assert numpy.allclose(N_l[1:-1], by_hand[1:-1])

    # phi_g = phi_g_inf, uniform:
    # N_g = (v*_g - delta' - (eta - 1) L') phi_g, with L' = -delta'
    phi_g_inf = numpy.array([0.3, 0.1])
    model.phi_g_ambient = phi_g_inf
    N_g = model.gas_faces(numpy.outer(phi_g_inf, numpy.ones_like(eta_g)), L_hat,
                          no_flux, v_star_g, ddelta_dt)
    by_hand = numpy.outer(phi_g_inf,
                          v_star_g - ddelta_dt + (faces_g - 1.0) * ddelta_dt)
    assert numpy.allclose(N_g[:, 1:], by_hand[:, 1:])

    # phi_g linear:
    # N_g = (v*_g - delta' - (eta - 1) L') phi_g + D_g (phi_g_s - phi_g_inf)/L_hat
    phi_g_s, phi_g_inf = numpy.array([0.5, 0.2]), numpy.array([0.1, 0.0])
    model.phi_g_ambient = phi_g_inf
    phi_g = (numpy.outer(phi_g_s, 2.0 - eta_g)
             + numpy.outer(phi_g_inf, eta_g - 1.0))
    N_g = model.gas_faces(phi_g, L_hat, no_flux, v_star_g, ddelta_dt)
    phi_g_f = (numpy.outer(phi_g_s, 2.0 - faces_g)
               + numpy.outer(phi_g_inf, faces_g - 1.0))
    by_hand = ((v_star_g - ddelta_dt + (faces_g - 1.0) * ddelta_dt) * phi_g_f
               + numpy.outer(D_g * (phi_g_s - phi_g_inf) / L_hat,
                             numpy.ones_like(faces_g)))
    assert numpy.allclose(N_g[:, 1:], by_hand[:, 1:])


def test_boundaries(model, delta_hat, liquid_mesh, gas_mesh):
    eta_l, _ = liquid_mesh
    eta_g, _ = gas_mesh

    # N_l = 0 at the substrate; N_l = n_A and N_g = n_i/theta_i at the surface,
    # n_i counting pure liquid and N_g gas volume
    n_i = numpy.array([0.7, 0.2])
    N_l = model.liquid_faces(liquid(0.2 + 0.6 * eta_l), delta_hat, n_i, -0.3)
    assert N_l[0, 0] == 0.0
    assert N_l[0, -1] == n_i[0]
    phi_g = numpy.outer(numpy.array([0.5, 0.2]), 2.0 - eta_g)
    N_g = model.gas_faces(phi_g, b_hat - delta_hat, n_i, 1.2, -0.3)
    assert numpy.array_equal(N_g[:, 0], n_i / theta)
