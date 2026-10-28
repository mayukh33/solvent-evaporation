import numpy
import scipy


>>>>>>> Stashed changes

class EvaporationModel:
    """One evaporating film: mixture groups, initial state, geometry, meshes."""

    def __init__(self, mixture, phi0, *, b_hat, liquid_grid, gas_grid,
                 phi_g_ambient=None):
        num_components = mixture.n
        phi0 = numpy.asarray(phi0, dtype=numpy.float64)

        self.mixture = mixture
        self.phi0 = phi0 / phi0.sum()
        # x0, the mole fractions of the pure liquids mixed in volume fractions phi0
        C0 = self.phi0 / mixture.V_bar
        self.x0 = C0 / C0.sum()
        self.b_hat = numpy.float64(b_hat)
        self.phi_g_ambient = (numpy.zeros(num_components) if phi_g_ambient is None
                              else numpy.asarray(phi_g_ambient, dtype=numpy.float64))
        self.liquid_grid = liquid_grid
        self.gas_grid = gas_grid

        self.liquid_end = (N - 1) * self.liquid_grid.n
        self.delta_index = self.liquid_end + N * self.gas_grid.n
        self.size = self.delta_index + 1

    # --- state ----------------------------------------------------------------

    def initial_state(self):
        """Uniform liquid, ambient gas, and initial delta_hat = 1.0.

        phi0 are the volume fractions of the pure liquids mixed to cast the
        film.  They fix its mole fractions x0, and the stored zeta follows:
        phi0 itself when volumes add.
        """
        m = self.mixture
        zeta0 = self.phi0
        if m.excess_volume is not None:
            zeta0 = self.x0 * m.V_bar / m.molar_volume(self.x0)[0]
        return self.pack(numpy.outer(zeta0[:-1], numpy.ones(self.liquid_grid.n)),
                         numpy.outer(self.phi_g_ambient, numpy.ones(self.gas_grid.n)),
                         1.0)

    def pack(self, zeta, phi_g, delta_hat):
        """The state from zeta (N-1, n_liquid), phi_g (N, n_gas) and delta_hat."""
        # stored u_i = delta_hat zeta_i, w_i = L_hat phi_g,i; L_hat = b_hat - delta_hat
        L_hat = self.b_hat - delta_hat
        return numpy.concatenate(((delta_hat * zeta).ravel(),
                                  (L_hat * phi_g).ravel(), [delta_hat]))

    def unpack(self, y):
        """As stored: zeta (N-1, n_liquid), phi_g (N, n_gas), delta_hat, L_hat."""
        # zeta_i = c_i v_i, the volume i would fill as pure liquid per unit volume
        # of film, is phi_i itself when volumes add
        num_components = self.mixture.n
        delta_hat = y[self.delta_index]
        L_hat = self.b_hat - delta_hat
        u = y[:self.liquid_end].reshape(num_components - 1, self.liquid_grid.n)
        w = y[self.liquid_end:self.delta_index].reshape(num_components,
                                                        self.gas_grid.n)
        return u / delta_hat, w / L_hat, delta_hat, L_hat

    def fields(self, y):
        """Unpack: phi (N, n_liquid), phi_g (N, n_gas), delta_hat, L_hat."""
        m = self.mixture
        zeta, phi_g, delta_hat, L_hat = self.unpack(y)
        phi = zeta
        if m.excess_volume is not None:
            # phi_i = nubar_i c_i, at the mole fractions each cell's zeta gives
            nu, c = self.liquid_cells(zeta)
            phi = (nu * c)[:-1]
        return (m.volume_fractions(dict(zip(m.components, phi))), phi_g, delta_hat,
                L_hat)

    # --- interface -------------------------------------------------------------
    def gas_side(self, phi_g_s, phi_g_first, L_hat):
        """The flux n_i and the draft s, given the vapour phi_g,i^s at the surface."""
        m = self.mixture
        # j_g,i = -(1/alpha_i) dphi_g,i/dz_hat -> (phi_g,i^s - phi_g,i,1) / (alpha_i h_g),
        # the interface half a cell from the first gas cell's centre
        h_g = 0.5 * L_hat * self.gas_grid.step
        j_gas = (phi_g_s - phi_g_first) / (m.alpha * h_g)
        # x_a^s = 1 - sum_i phi_g,i^s is the air the draft divides by
        vapour = phi_g_s.sum()
        if vapour >= 1.0:
            raise RuntimeError(
                f"interfacial vapour fills the gas (sum phi_g,i = {vapour:.4g}): "
                "the film is at or above its boiling point, which this model "
                "excludes.")
        # s = (sum_i j_g,i) / x_a^s: draft set by the insoluble air
        slip = j_gas.sum() / (1.0 - vapour)
        # n_i = theta_i (j_g,i + phi_g,i^s s): the gas's flux as volume of pure liquid
        return m.theta * (j_gas + phi_g_s * slip), slip

    def surface_root(self, balance, args, delta_hat, max_iter, tol):
        """The root of balance(root, *args) on the bracket [0, 1], checked.

        The check is on the balance times the liquid half cell h_l: the balance
        grows as 1/h_l while the film thins, and Brent's xtol alone leaves
        |d(balance)/d(root)| xtol of it at the root.
        """
        root = scipy.optimize.brentq(balance, 0.0, 1.0, args=args, xtol=1e-15,
                                     maxiter=max_iter)
        residual = 0.5 * delta_hat * self.liquid_grid.step * balance(root, *args)
        if abs(residual) > tol:
            raise RuntimeError(
                f"the interface solve stopped short of a root at {root}: "
                f"|r h_l| = {abs(residual):.3g}.")
        return root

    def jump_balance(self, phi_1s, phi_last, phi_g_first, delta_hat, L_hat,
                     full_output=False):
        """r = j_1 - phi_1^s delta' - n_1 at phi_1s: liquid side minus gas side.
        full_output: phi_s, the flux n_i, the draft s and delta'.
        """
        m = self.mixture
        phi_s = m.volume_fractions({m.components[0]: phi_1s})
        # phi_g,i^s = pi_i gamma_i x_i^(l)(phi^s): equilibrium at the interface
        flux, slip = self.gas_side(m.vapour_fraction(phi_s), phi_g_first, L_hat)
        ddelta_dt = -flux.sum()
        if full_output:
            return phi_s, flux, slip, ddelta_dt

        # j_1 = -D_AB(phibar) dphi_1/dz_hat -> D_AB (phi_1,nl - phi_1^s) / h_l, at
        # the mean of the two compositions the flux spans, half a cell apart
        h_l = 0.5 * delta_hat * self.liquid_grid.step
        phi_bar = m.volume_fractions({m.components[0]: 0.5 * (phi_1s + phi_last[0])})
        D_AB = m.fick_matrix(phi_bar)[0, 0]
        j_liquid = D_AB * (phi_last[0] - phi_1s) / h_l
        # r = j_1 - phi_1^s delta' - n_1 = 0: liquid side minus gas side
        return j_liquid - phi_1s * ddelta_dt - flux[0]

    def interface(self, phi_last, phi_g_first, delta_hat, L_hat, max_iter=100, tol=1e-12):
        """phi_s, the flux n_i, the draft s and delta' where the balance vanishes.

        The surface of a liquid whose volumes add; a non-ideal solution's is
        nonideal_surface.
        """
        args = (phi_last, phi_g_first, delta_hat, L_hat)
        phi_1s = self.surface_root(self.jump_balance, args, delta_hat, max_iter, tol)
        return self.jump_balance(phi_1s, *args, full_output=True)

    # --- interface of a non-ideal solution -------------------------------------
    def nonideal_jump_balance(self, x_1s, c_last, phi_g_first, delta_hat, L_hat,
                              full_output=False):
        """r = j_1 + phi_1^s (v*_s - delta') - (nubar_1,s/nu_1) n_1 at x_1s.

        The residual of a non-ideal solution in the surface mole fraction; c_last
        is the last liquid cell's c_i v_ref, (N,).  full_output: phi_s, the flux
        n_i, the draft s, x_s, nubar_s, the half-cell flux j_1 and v*_s - delta',
        the liquid's velocity relative to the surface.
        """
        m = self.mixture
        # the surface: nubar_i,s at x_s, c_i = x_i / V_m and phi_i = nubar_i c_i,
        # with V_m = sum_j x_j nubar_j
        x_s = numpy.array([x_1s, 1.0 - x_1s])
        nu_s = m.partial_volumes(x_s)
        c_s = x_s / numpy.sum(x_s * nu_s)
        phi_s = nu_s * c_s

        # the gas side as in an ideal solution: phi_g,i^s = pi_i gamma_i x_i^(l)
        flux, slip = self.gas_side(m.vapour_fraction_x(x_s), phi_g_first, L_hat)
        # v*_s - delta' = sum_i (nubar_i,s/nu_i) n_i: the volume the crossing
        # moles occupy at the surface composition
        relative = numpy.sum(nu_s / m.V_bar * flux)

        # j_1 = D_AB(xbar) nubar_1(xbar) (c_1,nl - c_1^s) / h_l, with xbar in
        # proportion to the mean of the two concentrations, half a cell apart
        h_l = 0.5 * delta_hat * self.liquid_grid.step
        x_bar = (c_s + c_last) / numpy.sum(c_s + c_last)
        j_liquid = (m.fick_matrix_x(x_bar)[0, 0] * m.partial_volumes(x_bar)[0]
                    * (c_last[0] - c_s[0]) / h_l)
        if full_output:
            return phi_s, flux, slip, x_s, nu_s, j_liquid, relative
        # r = j_1 + phi_1^s (v*_s - delta') - (nubar_1,s/nu_1) n_1 = 0: liquid
        # side minus gas side
        return j_liquid + phi_s[0] * relative - nu_s[0] / m.V_bar[0] * flux[0]

    def liquid_cells(self, zeta):
        """Each cell's nubar_i and c_i v_ref, each (2, n_liquid).

        One scalar solve per cell, for the mole fractions behind the stored
        zeta_A = x_A nu_A / V_m.
        """
        m = self.mixture
        x = m.mole_fractions_zeta(zeta[0])
        return m.partial_volumes(x), x / m.molar_volume(x)[0]

    def nonideal_surface(self, c_last, phi_g_first, delta_hat, L_hat, max_iter=100,
                         tol=1e-12):
        """nonideal_jump_balance's full output at its root in x_1s on [0, 1].

        c_last, the last liquid cell's c_i v_ref, is all the surface needs of
        the liquid.
        """
        args = (c_last, phi_g_first, delta_hat, L_hat)
        x_1s = self.surface_root(self.nonideal_jump_balance, args, delta_hat,
                                 max_iter, tol)
        return self.nonideal_jump_balance(x_1s, *args, full_output=True)

    def nonideal_faces(self, c, delta_hat):
        """Mean c_i, nubar_i and J_A on the interior liquid faces, (., n_liquid - 1).

        x_f is in proportion to the mean concentrations either side of face f,
        and J_A = -D_AB(x_f) dc_A/dz_hat across it.
        """
        m = self.mixture
        c_face = 0.5 * (c[:, :-1] + c[:, 1:])
        x_face = c_face / c_face.sum(axis=0)
        D_face = numpy.array([m.fick_matrix_x(x)[0, 0] for x in x_face.T])
        J_A = -D_face * numpy.diff(c[0]) / (delta_hat * self.liquid_grid.step)
        return c_face, m.partial_volumes(x_face), J_A

    def nonideal_interface(self, zeta, phi_g_first, delta_hat, L_hat):
        """The surface of a non-ideal solution, and delta' by v*_s, at the stored zeta.

        zeta is the liquid's zeta_A, (N-1, n_liquid), as unpack gives it.  The
        root of the residual fixes the surface, and delta' = v*_s - sum_i
        (nubar_i,s/nu_i) n_i follows, v*_s by the midpoint rule.  The march takes
        delta' from nonideal_liquid_faces instead, which this matches to second
        order.  Returns phi_s, the flux n_i, the draft s and delta', as
        interface does, then x_s and v*_s.
        """
        nu, c = self.liquid_cells(zeta)
        phi_s, flux, slip, x_s, nu_s, j_s, relative = self.nonideal_surface(
            c[:, -1], phi_g_first, delta_hat, L_hat)

        # v*_s by the midpoint rule on the cell centres: the flux nubar_A J_A on
        # each interior face times the change of ln(nubar_A/nubar_B) between the
        # cells either side of it, and the half-cell flux times its change up to
        # the surface.  The substrate's half cell carries no flux.
        _, nu_face, J_A = self.nonideal_faces(c, delta_hat)
        g = numpy.log(nu[0] / nu[1])
        v_star_s = (numpy.sum(nu_face[0] * J_A * numpy.diff(g))
                    + j_s * (numpy.log(nu_s[0] / nu_s[1]) - g[-1]))
        return phi_s, flux, slip, v_star_s - relative, x_s, v_star_s

    def nonideal_liquid_faces(self, nu, c, delta_hat, flux):
        """N_l on a non-ideal solution's liquid faces, (1, n_liquid + 1), and delta'.

        N_l,f = zetabar_f omega_f - D_AB(xbar_f) dzeta_A/dz_hat|_f, closed by no
        flux at the substrate and by n_A at the surface, with omega_f = v*_f -
        eta_f delta' the liquid's velocity past the moving face.  The volume
        closure, cell by cell, fixes omega_f and, in the top cell, delta'.
        nu and c are liquid_cells' nubar_i and c_i.
        """
        m = self.mixture
        liquid = self.liquid_grid

        # each interior face: the mean concentrations, J_A, and J_B =
        # -(nubar_A/nubar_B) J_A, so that sum_i nubar_i J_i = 0
        c_face, nu_face, J_A = self.nonideal_faces(c, delta_hat)
        J = numpy.array([J_A, -nu_face[0] / nu_face[1] * J_A])

        # sum_i nubar_i,c (N_i,c - N_i,c-1) = -delta' deta in each cell, with
        # N_i,f = c_i,f omega_f + J_i,f.  Face f is seen from the cell below
        # (lo) and the cell above (hi); marching up from the substrate, where
        # nothing crosses, is a lower bidiagonal system for omega = a + b delta'
        K_lo = numpy.sum(nu[:, :-1] * c_face, axis=0)
        K_hi = numpy.sum(nu[:, 1:] * c_face, axis=0)
        J_lo = numpy.sum(nu[:, :-1] * J, axis=0)
        J_hi = numpy.sum(nu[:, 1:] * J, axis=0)
        # the top cell, whose upper face is the surface, carrying N_i = n_i/nu_i
        into_gas = numpy.sum(nu[:, -1] * flux / m.V_bar)
        if liquid.n > 1:
            bands = numpy.array([K_lo, numpy.append(-K_hi[:-1], 0.0)])
            right = numpy.column_stack((numpy.append(0.0, J_hi[:-1]) - J_lo,
                                        numpy.full(liquid.n - 1, -liquid.step)))
            a, b = scipy.linalg.solve_banded((1, 0), bands, right).T
            ddelta_dt = ((K_hi[-1] * a[-1] + J_hi[-1] - into_gas)
                         / (liquid.step - K_hi[-1] * b[-1]))
        else:
            a = b = numpy.zeros(0)
            ddelta_dt = -into_gas / liquid.step
        omega = a + b * ddelta_dt

        # zeta_A's flux is nu_A times A's molar flux
        N_l = numpy.empty((1, liquid.n + 1))
        N_l[:, 0] = 0.0
        N_l[0, 1:-1] = m.V_bar[0] * (c_face[0] * omega + J_A)
        N_l[:, -1] = flux[0]
        return N_l, ddelta_dt

    # --- face fluxes -----------------------------------------------------------
    def liquid_faces(self, phi, delta_hat, flux, ddelta_dt):
        """N_l,i on the liquid faces f = 0 ... n_liquid, (N-1, n_liquid + 1).

        N_l,f = -[D](phibar_f) dphi/dz_hat|_f - eta_f delta' phibar_f, closed by no
        flux at the substrate and by the jump balance at the interface.
        """
        m = self.mixture
        n = m.n - 1                     # independent components
        liquid = self.liquid_grid

        # phibar_f = (phi_{f-1} + phi_f) / 2 on the interior faces, the balance
        # component included, and the diffusive flux -[D](phibar_f) dphi/dz_hat
        phi_face = 0.5 * (phi[:n, :-1] + phi[:n, 1:])
        components = {}
        for i, name in enumerate(m.components):
            components[name] = phi_face[i]
        face = m.volume_fractions(components)
        gradient = numpy.diff(phi[:n]) / (delta_hat * liquid.step)
        D = numpy.empty((liquid.n - 1, n, n))
        for f in range(liquid.n - 1):
            D[f] = m.fick_matrix(face[:, f])
        j = -numpy.einsum("fik,kf->if", D, gradient)

        N_l = numpy.empty((n, liquid.n + 1))
        N_l[:, 0] = 0.0
        # eta_f delta' phibar_f: the Landau term from the moving interface
        N_l[:, 1:-1] = j - liquid.faces[1:-1] * ddelta_dt * phi_face
        N_l[:, -1] = flux[:n]
        return N_l

    def gas_faces(self, phi_g, L_hat, flux, slip, ddelta_dt):
        """N_g,i on the gas faces f = 0 ... n_gas, (N, n_gas + 1), in gas volume.

        N_g,f = [s - (eta_f - 1) L'] phibar_g,f - (1/alpha_i) dphi_g/dz_hat|_f,
        closed by the jump balance at the interface and by the held ambient
        phi_g outside.
        """
        num_components = self.mixture.n
        alpha = self.mixture.alpha
        gas = self.gas_grid
        dL_dt = -ddelta_dt              # the outer edge is fixed in the laboratory

        # phibar_g,f and dphi_g/dz_hat on the interior faces
        phi_g_face = 0.5 * (phi_g[:, :-1] + phi_g[:, 1:])
        gradient = numpy.diff(phi_g) / (L_hat * gas.step)
        drift = (slip - (gas.faces[1:-1] - gas.start) * dL_dt) * phi_g_face

        N_g = numpy.empty((num_components, gas.n + 1))
        # n_i counts volume of pure liquid and the gas its own: n_i / theta_i
        N_g[:, 0] = flux / self.mixture.theta
        N_g[:, 1:-1] = drift - gradient / alpha[:, numpy.newaxis]
        # eta = 2: the same face flux, one-sided against the held ambient phi_g
        N_g[:, -1] = ((slip - dL_dt) * self.phi_g_ambient
                      - (self.phi_g_ambient - phi_g[:, -1])
                      / (alpha * 0.5 * L_hat * gas.step))
        return N_g

    # --- rate of change -------------------------------------------------------
    def liquid(self, zeta, phi_g_first, delta_hat, L_hat):
        """phi_s, the flux n_i, the draft s, delta' and the liquid's face fluxes N_l.

        zeta is the stored liquid over delta_hat.  When volumes add it is phi,
        and delta' comes with the surface from the jump balance; a non-ideal
        solution's is zeta_A, and delta' comes from the volume closure.
        """
        if self.mixture.excess_volume is None:
            phi_s, flux, slip, ddelta_dt = self.interface(zeta[:, -1], phi_g_first,
                                                          delta_hat, L_hat)
            N_l = self.liquid_faces(zeta, delta_hat, flux, ddelta_dt)
        else:
            nu, c = self.liquid_cells(zeta)
            phi_s, flux, slip = self.nonideal_surface(c[:, -1], phi_g_first,
                                                      delta_hat, L_hat)[:3]
            N_l, ddelta_dt = self.nonideal_liquid_faces(nu, c, delta_hat, flux)
        return phi_s, flux, slip, ddelta_dt, N_l

    def surface(self, y):
        """phi_s and the flux n_i at the state y."""
        zeta, phi_g, delta_hat, L_hat = self.unpack(y)
        return self.liquid(zeta, phi_g[:, 0], delta_hat, L_hat)[:2]

    def rhs(self, t, y):
        """d(state)/dt_hat.  t does not appear: nothing is driven externally."""
        zeta, phi_g, delta_hat, L_hat = self.unpack(y)
        # the surface, and beneath it d(delta_hat zeta_i)/dt_hat = -dN_l/deta
        _, flux, slip, ddelta_dt, N_l = self.liquid(zeta, phi_g[:, 0], delta_hat,
                                                    L_hat)
        # d(L_hat phi_g,i)/dt_hat = -dN_g/deta
        N_g = self.gas_faces(phi_g, L_hat, flux, slip, ddelta_dt)

        # du_c/dt_hat = -(N_l,c - N_l,c-1) / deta, and the same in the gas,
        # each stacked component by component as in the state; d(delta_hat)/dt_hat
        # = delta' closes it, so delta_hat takes the same implicit step
        du_dt = -numpy.diff(N_l) / self.liquid_grid.step
        dw_dt = -numpy.diff(N_g) / self.gas_grid.step
        return numpy.concatenate((du_dt.ravel(), dw_dt.ravel(), [ddelta_dt]))
