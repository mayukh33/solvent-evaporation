import numpy


class Ideal:
    def ln_gamma(self, x):
        return numpy.zeros_like(x)

    def thermodynamic_factor(self, x):
        return numpy.ones_like(x[0])

    def excess_volume(self, x):
        return 0.0, 0.0, 0.0


class Margules:
    def __init__(self, A12, A21):
        self.A12, self.A21 = float(A12), float(A21)

    def ln_gamma(self, x):
        x1, x2 = x
        return numpy.array([x2 ** 2 * (self.A12 + 2 * (self.A21 - self.A12) * x1),
                            x1 ** 2 * (self.A21 + 2 * (self.A12 - self.A21) * x2)])

    def thermodynamic_factor(self, x):
        x1, x2 = x
        return 1.0 - 2 * x1 * x2 * (self.A12 * (2 * x2 - x1) + self.A21 * (2 * x1 - x2))

class SoaveRedlichKwong:
    def __init__(self, R, T, P, v_ref, a, b, l=0.0, l_b=0.0):
        self.RT, self.P, self.v_ref = float(R) * float(T), float(P), float(v_ref)
        a, b = numpy.asarray(a, dtype=float), numpy.asarray(b, dtype=float)
        a_AB = (1.0 - l) * numpy.sqrt(a[0] * a[1])
        b_AB = (1.0 - l_b) * 0.5 * (b[0] + b[1])
        self.a_ij = numpy.array([[a[0], a_AB], [a_AB, a[1]]])
        self.b_ij = numpy.array([[b[0], b_AB], [b_AB, b[1]]])
        pure = numpy.eye(2)
        self.Z_pure = self.compressibility(pure)[0]
        self.G_R_pure = self.residual_gibbs(*self.mixing(pure)[:2], self.Z_pure)

    def mixing(self, x):
        x_A, x_B = x
        (a_AA, a_AB), (_, a_BB) = self.a_ij
        (b_AA, b_AB), (_, b_BB) = self.b_ij
        a = x_A ** 2 * a_AA + 2.0 * x_A * x_B * a_AB + x_B ** 2 * a_BB
        b = x_A ** 2 * b_AA + 2.0 * x_A * x_B * b_AB + x_B ** 2 * b_BB
        a_x = 2.0 * (x_A * a_AA + (x_B - x_A) * a_AB - x_B * a_BB)
        b_x = 2.0 * (x_A * b_AA + (x_B - x_A) * b_AB - x_B * b_BB)
        a_xx = 2.0 * (a_AA - 2.0 * a_AB + a_BB)
        b_xx = 2.0 * (b_AA - 2.0 * b_AB + b_BB)
        return a, b, a_x, b_x, a_xx, b_xx

    def root(self, a_tilde, beta):
        Z = beta
        for _ in range(100):
            C = Z ** 3 - Z ** 2 + (a_tilde - beta - beta ** 2) * Z - a_tilde * beta
            C_Z = 3.0 * Z ** 2 - 2.0 * Z + a_tilde - beta - beta ** 2
            step = C / C_Z
            Z = Z - step
            if (numpy.abs(step) <= 1e-14 * Z).all():
                if (Z >= 1.0 / 3.0).any():
                    raise RuntimeError("the Soave-Redlich-Kwong cubic has no liquid "
                                       "root below Z = 1/3")
                return Z
        raise RuntimeError("Newton's method for the liquid root of the "
                           "Soave-Redlich-Kwong cubic did not converge")

    def compressibility(self, x):
        a, b, a_x, b_x, a_xx, b_xx = self.mixing(x)
        to_a, to_b = self.P / self.RT ** 2, self.P / self.RT
        a_tilde, a_tilde_x, a_tilde_xx = a * to_a, a_x * to_a, a_xx * to_a
        beta, beta_x, beta_xx = b * to_b, b_x * to_b, b_xx * to_b
        Z = self.root(a_tilde, beta)
        C_Z = 3.0 * Z ** 2 - 2.0 * Z + a_tilde - beta - beta ** 2
        Z_x = (((1.0 + 2.0 * beta) * Z + a_tilde) * beta_x - (Z - beta) * a_tilde_x) / C_Z
        Z_xx = -((6.0 * Z - 2.0) * Z_x ** 2
                 + 2.0 * (a_tilde_x - (1.0 + 2.0 * beta) * beta_x) * Z_x
                 - 2.0 * beta_x * (a_tilde_x + Z * beta_x)
                 + (Z - beta) * a_tilde_xx
                 - ((1.0 + 2.0 * beta) * Z + a_tilde) * beta_xx) / C_Z
        return Z, Z_x, Z_xx

    def residual_gibbs(self, a, b, Z):
        beta = b * self.P / self.RT
        return Z - 1.0 - numpy.log(Z - beta) - a / (b * self.RT) * numpy.log1p(beta / Z)

    def excess_volume(self, x):
        Z, Z_x, Z_xx = self.compressibility(x)
        to_V = self.RT / (self.P * self.v_ref)
        return (to_V * (Z - x[0] * self.Z_pure[0] - x[1] * self.Z_pure[1]),
                to_V * (Z_x - self.Z_pure[0] + self.Z_pure[1]),
                to_V * Z_xx)

    def excess_gibbs(self, x):
        a, b, a_x, b_x, a_xx, b_xx = self.mixing(x)
        Z, Z_x, _ = self.compressibility(x)
        xi = b * self.P / (self.RT * Z)
        ratio, ln_1_xi = a / (b * self.RT), numpy.log1p(xi)
        A_x, B_x = a_x / a, b_x / b
        G = self.residual_gibbs(a, b, Z) - x[0] * self.G_R_pure[0] - x[1] * self.G_R_pure[1]
        G_x = ((Z - 1.0) * B_x - ratio * ln_1_xi * (A_x - B_x)
               - (self.G_R_pure[0] - self.G_R_pure[1]))
        G_xx = (Z_x * B_x + (Z - 1.0) * (b_xx / b - B_x ** 2)
                - ratio * ln_1_xi * (a_xx / a - b_xx / b - 2.0 * B_x * (A_x - B_x))
                - ratio * xi / (1.0 + xi) * (B_x - Z_x / Z) * (A_x - B_x))
        return G, G_x, G_xx

    def ln_gamma(self, x):
        G, G_x, _ = self.excess_gibbs(x)
        return numpy.array([G + x[1] * G_x, G - x[0] * G_x])

    def thermodynamic_factor(self, x):
        _, _, G_xx = self.excess_gibbs(x)
        return 1.0 + x[0] * x[1] * G_xx


class Mixture:
    def __init__(self, phi_g_sat, alpha, theta, V_bar, D0, excess_volume,
                 activity=None):
        self.phi_g_sat = numpy.asarray(phi_g_sat, dtype=float)
        self.alpha = numpy.asarray(alpha, dtype=float)
        self.theta = numpy.asarray(theta, dtype=float)
        self.V_bar = numpy.asarray(V_bar, dtype=float)
        self.D0 = numpy.asarray(D0, dtype=float)
        if self.phi_g_sat.size != 2:
            raise ValueError(f"Give two solvents, not {self.phi_g_sat.size}")
        self.activity = Ideal() if activity is None else activity
        self.excess_volume = Ideal() if excess_volume is None else excess_volume

    def molar_volume(self, x):
        V = x[0] * self.V_bar[0] + x[1] * self.V_bar[1]
        dV = self.V_bar[0] - self.V_bar[1]
        V_E, dV_E, d2V_E = self.excess_volume.excess_volume(x)
        return V + V_E, dV + dV_E, d2V_E

    def partial_volumes(self, x):
        V, dV, _ = self.molar_volume(x)
        return numpy.array([V + x[1] * dV, V - x[0] * dV])

    def mole_fractions_phi(self, phi_A, x_A):
        phi_A = numpy.asarray(phi_A, dtype=float)
        x_A = numpy.array(x_A, dtype=float)
        for _ in range(100):
            x = numpy.array([x_A, 1.0 - x_A])
            V, _, d2V = self.molar_volume(x)
            nu_A, nu_B = self.partial_volumes(x)
            slope = (nu_A * nu_B + x_A * (1.0 - x_A) * V * d2V) / V ** 2
            step = (x_A * nu_A / V - phi_A) / slope
            x_A = x_A - step
            if numpy.abs(step).max() < 1e-15:
                return numpy.array([x_A, 1.0 - x_A])
        raise RuntimeError(f"Newton's method for x_A did not converge at phi_A = {phi_A}")

    def s_x(self, x):
        V, _, d2V = self.molar_volume(x)
        nu_A, nu_B = self.partial_volumes(x)
        curvature = x[0] * x[1] * V * d2V
        return curvature / (nu_A * nu_B + curvature)

    def vapour_fraction_x(self, x):
        return self.phi_g_sat * numpy.exp(self.activity.ln_gamma(x)) * x

    def diffusivity_x(self, x):
        D_MS = self.D0[1] ** x[0] * self.D0[0] ** x[1]
        return D_MS * self.activity.thermodynamic_factor(x)
