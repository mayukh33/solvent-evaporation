import numpy
import scipy

# njit comes from diffusion so the optional-numba fallback lives in one place.
from .diffusion import fick_matrix, njit


@njit(cache=True)
def liquid_face_fluxes(phi, phi_face, Gamma, nu, D0, delta_hat, flux,
                       ddelta_dt, faces, step):
    """F_i on the liquid faces f = 0 ... n_liquid, at [Gamma] given per face.

    F_f = -[D](phibar_f) dphi/dz_hat|_f - eta_f delta' phibar_f, closed by no
    flux at the substrate and by the jump balance at the interface.
    """
    N = phi.shape[0]
    n = N - 1
    nl = phi.shape[1]

    F = numpy.empty((n, nl + 1))
    for i in range(n):
        F[i, 0] = 0.0
        F[i, nl] = flux[i]
    for f in range(1, nl):
        D = fick_matrix(phi_face[f - 1], nu, D0, Gamma[f - 1])
        for i in range(n):
            diffusion = 0.0
            for k in range(n):
                diffusion += D[i, k] * ((phi[k, f] - phi[k, f - 1])
                                        / (delta_hat * step))
            # eta_f delta' phibar_f: the Landau term from the moving interface
            F[i, f] = -diffusion - faces[f] * ddelta_dt * phi_face[f - 1, i]
    return F


@njit(cache=True)
def gas_face_fluxes(psi, L_hat, flux, slip, ddelta_dt, alpha, psi_ambient,
                    faces, start, step):
    """G_i on the gas faces f = 0 ... n_gas.

    G_f = [s - (eta_f - 1) L'] psibar_f - (1/alpha_i) dpsi/dz_hat|_f, closed by
    the jump balance at the interface and by the held ambient psi outside.
    """
    N = psi.shape[0]
    ng = psi.shape[1]
    dL_dt = -ddelta_dt              # the outer edge is fixed in the laboratory

    G = numpy.empty((N, ng + 1))
    for i in range(N):
        G[i, 0] = flux[i]
        for f in range(1, ng):
            psi_face = 0.5 * (psi[i, f - 1] + psi[i, f])
            drift = (slip - (faces[f] - start) * dL_dt) * psi_face
            gradient = (psi[i, f] - psi[i, f - 1]) / (L_hat * step)
            G[i, f] = drift - gradient / alpha[i]
        # eta = 2: the same face flux, one-sided against the held ambient psi
        G[i, ng] = ((slip - dL_dt) * psi_ambient[i]
                    - (psi_ambient[i] - psi[i, ng - 1])
                    / (alpha[i] * 0.5 * L_hat * step))
    return G


class EvaporationModel:
    """One evaporating film: mixture groups, initial state, geometry, meshes."""

    def __init__(self, mixture, phi0, *, b_hat, liquid_grid, gas_grid,
                 psi_ambient=None):
        N = mixture.n
        phi0 = numpy.asarray(phi0, dtype=numpy.float64)

        self.mixture = mixture
        self.phi0 = phi0 / phi0.sum()
        self.b_hat = numpy.float64(b_hat)
        self.L_hat0 = self.b_hat - 1.0
        self.psi_ambient = (numpy.zeros(N) if psi_ambient is None
                            else numpy.asarray(psi_ambient, dtype=numpy.float64))
        self.liquid_grid = liquid_grid
        self.gas_grid = gas_grid

        self.liquid_end = (N - 1) * self.liquid_grid.n
        self.delta_index = self.liquid_end + N * self.gas_grid.n
        self.size = self.delta_index + 1

    # --- state ----------------------------------------------------------------

    def initial_state(self):
        """Uniform liquid, ambient gas, delta_hat = 1."""
        nl, ng = self.liquid_grid.n, self.gas_grid.n
        return numpy.concatenate((
            numpy.repeat(self.phi0[:-1], nl),
            numpy.repeat(self.psi_ambient, ng) * self.L_hat0,
            [1.0]))

    def fields(self, y):
        """Unpack: phi (N, n_liquid), psi (N, n_gas), delta_hat, L_hat."""
        # stored u_i = delta_hat phi_i, w_i = L_hat psi_i; L_hat = b_hat - delta_hat
        N = self.mixture.n
        delta_hat = y[self.delta_index]
        L_hat = self.b_hat - delta_hat
        u = y[:self.liquid_end].reshape(N - 1, self.liquid_grid.n)
        w = y[self.liquid_end:self.delta_index].reshape(N, self.gas_grid.n)
        components = {}
        for i, name in enumerate(self.mixture.components):
            components[name] = u[i] / delta_hat
        phi = self.mixture.volume_fractions(components)
        return phi, w / L_hat, delta_hat, L_hat

    # --- interface -------------------------------------------------------------
    def jump_balance(self, phi_1s, phi_last, psi_first, delta_hat, L_hat,
                     full_output=False):
        """r = j_1 - phi_1^s delta' - n_1 at phi_1s: liquid side minus gas side.
        full_output: phi_s, the flux n_i, the draft s and delta'.
        """
        m = self.mixture
        # the interface lies half a cell from either neighbouring cell centre
        h_l = 0.5 * delta_hat * self.liquid_grid.step
        h_g = 0.5 * L_hat * self.gas_grid.step
        phi_s = m.volume_fractions({m.components[0]: phi_1s})

        # psi_i^s = gamma_i gamma^act_i x_i^(l)(phi^s): equilibrium at the interface
        psi_s = m.vapour_fraction(phi_s)
        # jt_i = -(1/alpha_i) dpsi_i/dz_hat -> (psi_i^s - psi_i,1) / (alpha_i h_g)
        j_gas = (psi_s - psi_first) / (m.alpha * h_g)
        # xg_i = psi_i/theta_i are the gas mole fractions, so x_a^s = 1 - sum_i xg_i^s
        vapour = (psi_s / m.theta).sum()
        if vapour >= 1.0:
            raise RuntimeError(
                f"interfacial vapour fills the gas (sum xg_i = {vapour:.4g}): "
                "the film is at or above its boiling point, which this model "
                "excludes.")
        # s = (sum_i jt_i/theta_i) / x_a^s: draft set by the insoluble air
        slip = (j_gas / m.theta).sum() / (1.0 - vapour)
        flux = psi_s * slip + j_gas
        ddelta_dt = -flux.sum()
        if full_output:
            return phi_s, flux, slip, ddelta_dt

        # j_1 = -D(phibar) dphi_1/dz_hat -> D (phi_1,nl - phi_1^s) / h_l
        D = m.fick_matrix(0.5 * (phi_s + phi_last))[0, 0]
        j_liquid = D * (phi_last[0] - phi_s[0]) / h_l
        # r = j_1 - phi_1^s delta' - n_1 = 0: liquid side minus gas side
        return j_liquid - phi_s[0] * ddelta_dt - flux[0]

    def interface(self, phi_last, psi_first, delta_hat, L_hat, max_iter=100, tol=1e-12):
        """phi_s, the flux n_i, the draft s and delta' where the balance vanishes."""
        cells = (phi_last, psi_first, delta_hat, L_hat)
        phi_1s = scipy.optimize.brentq(self.jump_balance, 0.0, 1.0, args=cells,
                                       xtol=1e-15, maxiter=max_iter)
        residual = self.jump_balance(phi_1s, *cells)
        if abs(residual) > tol:
            raise RuntimeError(
                f"the interface solve stopped short of a root at phi_s = "
                f"{phi_1s}: |r| = {abs(residual):.3g}.")
        return self.jump_balance(phi_1s, *cells, full_output=True)

    # --- face fluxes -----------------------------------------------------------
    def liquid_faces(self, phi, delta_hat, flux, ddelta_dt):
        """F_i on the liquid faces, by liquid_face_fluxes.

        The face compositions and their [Gamma] are built here, because the
        activity model is a Python object and cannot enter the jitted loop.
        """
        m = self.mixture
        N, n = m.n, m.n - 1
        liquid = self.liquid_grid

        phi_face = numpy.empty((liquid.n - 1, N))
        Gamma = numpy.empty((liquid.n - 1, n, n))
        for f in range(1, liquid.n):
            for k in range(N):
                # phibar_f = (phi_{f-1} + phi_f) / 2
                phi_face[f - 1, k] = 0.5 * (phi[k, f - 1] + phi[k, f])
            Gamma[f - 1] = m.activity.thermodynamic_factor(
                m.mole_fractions(phi_face[f - 1]))
        return liquid_face_fluxes(phi, phi_face, Gamma, m.nu, m.D0, delta_hat,
                                  flux, ddelta_dt, liquid.faces, liquid.step)

    def gas_faces(self, psi, L_hat, flux, slip, ddelta_dt):
        """G_i on the gas faces, by gas_face_fluxes."""
        gas = self.gas_grid
        return gas_face_fluxes(psi, L_hat, flux, slip, ddelta_dt,
                               self.mixture.alpha, self.psi_ambient,
                               gas.faces, gas.start, gas.step)

    def rhs(self, t, y):
        """d(state)/dt_hat.  t does not appear: nothing is driven externally."""
        N, n = self.mixture.n, self.mixture.n - 1
        liquid, gas = self.liquid_grid, self.gas_grid
        phi, psi, delta_hat, L_hat = self.fields(y)
        _, flux, slip, ddelta_dt = self.interface(phi[:, -1], psi[:, 0],
                                                 delta_hat, L_hat)
        # d(delta_hat phi_i)/dt_hat = -dF/deta, d(L_hat psi_i)/dt_hat = -dG/deta
        F = self.liquid_faces(phi, delta_hat, flux, ddelta_dt)
        G = self.gas_faces(psi, L_hat, flux, slip, ddelta_dt)

        # du_c/dt_hat = -(F_c - F_{c-1}) / deta, and the same in the gas
        dy = numpy.empty(self.size)
        for i in range(n):
            for c in range(liquid.n):
                dy[i * liquid.n + c] = -(F[i, c + 1] - F[i, c]) / liquid.step
        for i in range(N):
            for c in range(gas.n):
                dy[self.liquid_end + i * gas.n + c] = (-(G[i, c + 1] - G[i, c])
                                                       / gas.step)
        dy[self.delta_index] = ddelta_dt
        return dy
