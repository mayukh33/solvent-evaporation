import numpy
import scipy


class EvaporationModel:
    """One evaporating film: mixture groups, initial state, geometry, meshes."""

    def __init__(self, mixture, phi0, *, b_hat, liquid_grid, gas_grid,
                 phi_g_ambient=None):
        num_components = mixture.n
        phi0 = numpy.asarray(phi0, dtype=numpy.float64)

        self.mixture = mixture
        self.phi0 = phi0 / phi0.sum()
        self.b_hat = numpy.float64(b_hat)
        self.L_hat0 = self.b_hat - 1.0
        self.phi_g_ambient = (numpy.zeros(num_components) if phi_g_ambient is None
                              else numpy.asarray(phi_g_ambient, dtype=numpy.float64))
        self.liquid_grid = liquid_grid
        self.gas_grid = gas_grid

        self.liquid_end = (num_components - 1) * self.liquid_grid.n
        self.delta_index = self.liquid_end + num_components * self.gas_grid.n
        self.size = self.delta_index + 1

    # --- state ----------------------------------------------------------------

    def initial_state(self):
        """Uniform liquid, ambient gas, and initial delta_hat = 1.0."""
        nl, ng = self.liquid_grid.n, self.gas_grid.n
        delta_hat0 = 1.0
        y0 = numpy.concatenate((
            numpy.repeat(self.phi0[:-1], nl),
            numpy.repeat(self.phi_g_ambient, ng) * self.L_hat0,
            [delta_hat0]))
        return y0

    def fields(self, y):
        """Unpack: phi (N, n_liquid), phi_g (N, n_gas), delta_hat, L_hat."""
        # stored u_i = delta_hat phi_i, w_i = L_hat phi_g_i; L_hat = b_hat - delta_hat
        num_components = self.mixture.n
        delta_hat = y[self.delta_index]
        L_hat = self.b_hat - delta_hat
        u = y[:self.liquid_end].reshape(num_components - 1, self.liquid_grid.n)
        w = y[self.liquid_end:self.delta_index].reshape(num_components,
                                                        self.gas_grid.n)
        components = {}
        for i, name in enumerate(self.mixture.components):
            components[name] = u[i] / delta_hat
        phi = self.mixture.volume_fractions(components)
        return phi, w / L_hat, delta_hat, L_hat

    # --- interface -------------------------------------------------------------
    def jump_balance(self, phi_1s, phi_last, phi_g_first, delta_hat, L_hat,
                     full_output=False):
        """r = j_1 - phi_1^s delta' - n_1 at phi_1s: liquid side minus gas side.
        full_output: phi_s, the flux n_i, the gas velocity v*_g and delta'.
        """
        m = self.mixture
        # the interface lies half a cell from either neighbouring cell centre
        h_l = 0.5 * delta_hat * self.liquid_grid.step
        h_g = 0.5 * L_hat * self.gas_grid.step
        phi_s = m.volume_fractions({m.components[0]: phi_1s})

        # phi_g_i^s = phi_g_sat,i gamma_i x_i^(l)(phi^s): equilibrium at the interface
        phi_g_s = m.vapour_fraction(phi_s)
        # j_i = -(1/alpha_i) dphi_g_i/dz_hat -> (phi_g_i^s - phi_g_i,1) / (alpha_i h_g)
        j_gas = (phi_g_s - phi_g_first) / (m.alpha * h_g)
        # x_a^s = 1 - sum_i phi_g_i^s: the air left at the interface
        vapour = phi_g_s.sum()
        if vapour >= 1.0:
            raise RuntimeError(
                f"interfacial vapour fills the gas (sum phi_g_i = {vapour:.4g}): "
                "the film is at or above its boiling point, which this model "
                "excludes.")
        # n_i = theta_i (j_i + phi_g_i^s (sum_k j_k)/x_a^s): the gas side as volume
        # of pure liquid, the fraction being the draft v*_g - delta' that the
        # insoluble air sets; theta_i = v_i P/(R T) is pure liquid i's molar volume
        # over the gas's
        gas_flux = m.theta * (j_gas + phi_g_s * j_gas.sum() / (1.0 - vapour))
        ddelta_dt = -gas_flux.sum()
        if full_output:
            # v*_g = delta' + (sum_i j_i)/x_a^s: the gas's volume-average velocity
            v_star_g = ddelta_dt + j_gas.sum() / (1.0 - vapour)
            return phi_s, gas_flux, v_star_g, ddelta_dt

        D_AB = m.fick_matrix(0.5 * (phi_s + phi_last))[0, 0]
        j_liquid = D_AB * (phi_last[0] - phi_s[0]) / h_l
        # r = j_1 - phi_1^s delta' - n_1 = 0: liquid side minus gas side
        return j_liquid - phi_s[0] * ddelta_dt - gas_flux[0]

    def interface(self, phi_last, phi_g_first, delta_hat, L_hat, max_iter=100, tol=1e-12):
        """phi_s, the flux n_i, the gas velocity v*_g and delta' where the balance
        vanishes."""
        cells = (phi_last, phi_g_first, delta_hat, L_hat)
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
        """N_l,i on the liquid faces f = 0 ... n_liquid, (N-1, n_liquid + 1).

        N_l,f = -[D](phibar_f) dphi/dz_hat|_f - eta_f delta' phibar_f, closed by no
        flux at the substrate and by the jump balance at the interface.
        """
        m = self.mixture
        n = m.n - 1                     # independent components
        liquid = self.liquid_grid

        # phibar_f = (phi_{f-1} + phi_f) / 2 and dphi/dz_hat on the interior faces
        phi_face = 0.5 * (phi[:, :-1] + phi[:, 1:])
        gradient = numpy.diff(phi[:n]) / (delta_hat * liquid.step)
        D = numpy.empty((liquid.n - 1, n, n))
        for f in range(liquid.n - 1):
            D[f] = m.fick_matrix(phi_face[:, f])
        # [D] dphi/dz_hat on each face f: sum_k D_ik,f dphi_k/dz_hat|_f
        diffusion = numpy.einsum("fik,kf->if", D, gradient)
        # eta_f delta' phibar_f: the Landau term from the moving interface
        landau = liquid.faces[1:-1] * ddelta_dt * phi_face[:n]

        N_l = numpy.empty((n, liquid.n + 1))
        N_l[:, 0] = 0.0
        N_l[:, 1:-1] = -diffusion - landau
        N_l[:, -1] = flux[:n]
        return N_l

    def gas_faces(self, phi_g, L_hat, flux, v_star_g, ddelta_dt):
        """N_g,i on the gas faces f = 0 ... n_gas, (N, n_gas + 1), in gas volume.

        N_g,f = [v*_g - delta' - (eta_f - 1) L'] phibar_g,f
        - (1/alpha_i) dphi_g/dz_hat|_f, closed by the jump balance at the interface
        and by the held ambient phi_g outside.
        """
        num_components = self.mixture.n
        alpha = self.mixture.alpha
        theta = self.mixture.theta
        gas = self.gas_grid
        dL_dt = -ddelta_dt              # the outer edge is fixed in the laboratory

        # phibar_g,f and dphi_g/dz_hat on the interior faces
        phi_g_face = 0.5 * (phi_g[:, :-1] + phi_g[:, 1:])
        gradient = numpy.diff(phi_g) / (L_hat * gas.step)
        drift = ((v_star_g - ddelta_dt - (gas.faces[1:-1] - gas.start) * dL_dt)
                 * phi_g_face)

        N_g = numpy.empty((num_components, gas.n + 1))
        # eta = 1: the jump balance's n_i over theta_i, from pure liquid to gas volume
        N_g[:, 0] = flux / theta
        N_g[:, 1:-1] = drift - gradient / alpha[:, numpy.newaxis]
        # eta = 2: the same face flux, one-sided against the held ambient phi_g;
        # the edge is fixed, so the drift there is v*_g itself
        N_g[:, -1] = (v_star_g * self.phi_g_ambient
                      - (self.phi_g_ambient - phi_g[:, -1])
                      / (alpha * 0.5 * L_hat * gas.step))
        return N_g

    # --- rate of change -------------------------------------------------------
    def rhs(self, t, y):
        """d(state)/dt_hat.  t does not appear: nothing is driven externally."""
        liquid, gas = self.liquid_grid, self.gas_grid
        phi, phi_g, delta_hat, L_hat = self.fields(y)
        _, flux, v_star_g, ddelta_dt = self.interface(phi[:, -1], phi_g[:, 0],
                                                      delta_hat, L_hat)

        # d(delta_hat phi_i)/dt_hat = -dN_l/deta, d(L_hat phi_g_i)/dt_hat = -dN_g/deta
        N_l = self.liquid_faces(phi, delta_hat, flux, ddelta_dt)
        N_g = self.gas_faces(phi_g, L_hat, flux, v_star_g, ddelta_dt)

        # du_c/dt_hat = -(N_l,c - N_l,c-1) / deta, and the same in the gas,
        # each stacked component by component as in the state; d(delta_hat)/dt_hat
        # = delta' closes it, so delta_hat takes the same implicit step
        du_dt = -numpy.diff(N_l) / liquid.step
        dw_dt = -numpy.diff(N_g) / gas.step
        return numpy.concatenate((du_dt.ravel(), dw_dt.ravel(), [ddelta_dt]))