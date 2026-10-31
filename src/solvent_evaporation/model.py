import numpy



class EvaporationModel:
    def __init__(self, mixture, phi0, *, b_hat, liquid_grid, gas_grid,
                 phi_g_ambient=None):
        num_components = mixture.n
        phi0 = numpy.asarray(phi0, dtype=numpy.float64)

        self.mixture = mixture
        self.phi0 = phi0 / phi0.sum()
        C0 = self.phi0 / mixture.V_bar
        self.x0 = C0 / C0.sum()
        self.b_hat = numpy.float64(b_hat)
        self.phi_g_ambient = (numpy.zeros(num_components) if phi_g_ambient is None
                              else numpy.asarray(phi_g_ambient, dtype=numpy.float64))
        self.liquid_grid = liquid_grid
        self.gas_grid = gas_grid
        self.x_nodes = numpy.full(self.liquid_grid.n + 1, self.x0[0])
        self.x_faces = numpy.full(self.liquid_grid.n, self.x0[0])

        self.liquid_end = (num_components - 1) * (self.liquid_grid.n + 1)
        self.delta_index = self.liquid_end + num_components * (self.gas_grid.n - 1)
        self.size = self.delta_index + 1

    def initial_state(self):
        m = self.mixture
        phi_cast = self.x0 * m.partial_volumes(self.x0) / m.molar_volume(self.x0)[0]
        self.x_nodes = numpy.full(self.liquid_grid.n + 1, self.x0[0])
        self.x_faces = numpy.full(self.liquid_grid.n, self.x0[0])
        return self.pack(numpy.outer(phi_cast[:-1], numpy.ones(self.liquid_grid.n + 1)),
                         numpy.outer(self.phi_g_ambient, numpy.ones(self.gas_grid.n - 1)),
                         1.0)

    def pack(self, phi, phi_g, delta_hat):
        L_hat = self.b_hat - delta_hat
        return numpy.concatenate((numpy.asarray(phi, dtype=numpy.float64).ravel(),
                                  (L_hat * phi_g).ravel(), [delta_hat]))

    def unpack(self, y):
        num_components = self.mixture.n
        delta_hat = y[self.delta_index]
        L_hat = self.b_hat - delta_hat
        phi = y[:self.liquid_end].reshape(num_components - 1, self.liquid_grid.n + 1)
        w = y[self.liquid_end:self.delta_index].reshape(num_components,
                                                        self.gas_grid.n - 1)
        return phi, w / L_hat, delta_hat, L_hat

    def fields(self, y):
        m = self.mixture
        phi, phi_g, delta_hat, L_hat = self.unpack(y)
        x = self.liquid_nodes(phi)[0]
        phi_g_node = numpy.column_stack((m.vapour_fraction_x(x[:, -1]), phi_g,
                                         self.phi_g_ambient))
        return (m.volume_fractions(dict(zip(m.components, phi))), phi_g_node,
                delta_hat, L_hat)

    def gas_side(self, phi_g_s, phi_g_first, L_hat):
        m = self.mixture
        h_g = L_hat * self.gas_grid.step
        j_gas = (phi_g_s - phi_g_first) / (m.alpha * h_g)
        vapour = phi_g_s.sum()
        if vapour >= 1.0:
            raise RuntimeError(
                f"interfacial vapour fills the gas (sum phi_g,i = {vapour:.4g}): "
                "the film is at or above its boiling point, which this model "
                "excludes.")
        slip = j_gas.sum() / (1.0 - vapour)
        return m.theta * (j_gas + phi_g_s * slip), slip

    def liquid_nodes(self, phi):
        m = self.mixture
        x = m.mole_fractions_phi(phi[0, :], self.x_nodes)
        self.x_nodes = x[0, :]
        D = numpy.array([m.fick_matrix_x(x_m)[0, 0] for x_m in x.T])
        return x, m.partial_volumes(x), m.s_x(x), D

    def liquid_faces(self, phi, delta_hat):
        m = self.mixture
        phi_face = 0.5 * (phi[0, :-1] + phi[0, 1:])
        x = m.mole_fractions_phi(phi_face, self.x_faces)
        self.x_faces = x[0, :]
        s = m.s_x(x)
        D = numpy.array([m.fick_matrix_x(x_f)[0, 0] for x_f in x.T])
        j_A = -D / delta_hat * (1.0 - s) * numpy.diff(phi[0, :]) / self.liquid_grid.step
        return phi_face, s, j_A

    def liquid_surface(self, x_s, phi_s, nu_s, s_s, D_s, phi_g_first, delta_hat,
                       L_hat):
        m = self.mixture
        flux, slip = self.gas_side(m.vapour_fraction_x(x_s), phi_g_first, L_hat)
        relative = numpy.sum(nu_s / m.V_bar * flux)
        F_s = nu_s[0] / m.V_bar[0] * flux[0]
        j_s = F_s - phi_s * relative
        dphi_s = -delta_hat * j_s / (D_s * (1.0 - s_s))
        return flux, slip, relative, F_s, j_s, dphi_s

    def liquid_rates(self, phi, s, phi_face, s_face, j_face, delta_hat, relative,
                     F_s, j_s, dphi_s):
        grid = self.liquid_grid
        d_eta = grid.step
        phi_A = phi[0, :]

        increment = s_face * j_face / (phi_face * (1.0 - phi_face)) * numpy.diff(phi_A)
        v_star = numpy.append(0.0, numpy.cumsum(increment))
        v_star_face = 0.5 * (v_star[:-1] + v_star[1:])
        ddelta_dt = v_star[-1] - relative

        F = numpy.concatenate(([0.0],
                               j_face + (v_star_face - grid.centers * ddelta_dt) * phi_face,
                               [F_s]))
        j_node = numpy.concatenate(([0.0], 0.5 * (j_face[:-1] + j_face[1:]), [j_s]))
        dphi = numpy.concatenate(([0.0], (phi_A[2:] - phi_A[:-2]) / (2.0 * d_eta),
                                  [dphi_s]))
        u_A = v_star + j_node / phi_A
        width = numpy.full(grid.n + 1, d_eta)
        width[[0, -1]] = 0.5 * d_eta

        dphi_dt = ((-ddelta_dt * phi_A - numpy.diff(F) / width
                    + s * (u_A - grid.faces * ddelta_dt) * dphi)
                   / ((1.0 - s) * delta_hat))
        return dphi_dt[numpy.newaxis, :], ddelta_dt

    def liquid(self, phi, phi_g_first, delta_hat, L_hat):
        m = self.mixture
        x, nu, s, D = self.liquid_nodes(phi)
        phi_face, s_face, j_face = self.liquid_faces(phi, delta_hat)
        flux, slip, relative, F_s, j_s, dphi_s = self.liquid_surface(
            x[:, -1], phi[0, -1], nu[:, -1], s[-1], D[-1], phi_g_first, delta_hat,
            L_hat)
        dphi_dt, ddelta_dt = self.liquid_rates(phi, s, phi_face, s_face, j_face,
                                               delta_hat, relative, F_s, j_s, dphi_s)
        phi_s = m.volume_fractions(dict(zip(m.components, phi[:, -1])))
        return phi_s, flux, slip, ddelta_dt, dphi_dt

    def surface(self, y):
        phi, phi_g, delta_hat, L_hat = self.unpack(y)
        return self.liquid(phi, phi_g[:, 0], delta_hat, L_hat)[:2]

    def gas_faces(self, phi_g, L_hat, flux, slip, ddelta_dt):
        num_components = self.mixture.n
        alpha = self.mixture.alpha
        gas = self.gas_grid
        dL_dt = -ddelta_dt

        phi_g_node = numpy.column_stack((phi_g, self.phi_g_ambient))
        phi_g_face = 0.5 * (phi_g_node[:, :-1] + phi_g_node[:, 1:])
        gradient = numpy.diff(phi_g_node) / (L_hat * gas.step)
        drift = (slip - (gas.centers[1:] - gas.start) * dL_dt) * phi_g_face

        N_g = numpy.empty((num_components, gas.n))
        N_g[:, 0] = flux / self.mixture.theta
        N_g[:, 1:] = drift - gradient / alpha[:, numpy.newaxis]
        return N_g

    def rhs(self, t, y):
        phi, phi_g, delta_hat, L_hat = self.unpack(y)
        _, flux, slip, ddelta_dt, dphi_dt = self.liquid(phi, phi_g[:, 0], delta_hat,
                                                        L_hat)
        N_g = self.gas_faces(phi_g, L_hat, flux, slip, ddelta_dt)

        dw_dt = -numpy.diff(N_g) / self.gas_grid.step
        return numpy.concatenate((dphi_dt.ravel(), dw_dt.ravel(), [ddelta_dt]))
