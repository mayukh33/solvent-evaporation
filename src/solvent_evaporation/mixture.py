import numpy
import scipy

from .cache import composition_cache


class Ideal:
    def ln_gamma(self, x):
        return numpy.zeros_like(x)

    def thermodynamic_factor(self, x):
        return numpy.ones_like(x[0])

    def excess_volume(self, x):
        return 0.0, 0.0, 0.0


class Margules:
    def __init__(self, A_12, A_21):
        self.A_12, self.A_21 = float(A_12), float(A_21)

    def ln_gamma(self, x):
        x_A, x_B = x
        return numpy.array([x_B ** 2 * (self.A_12 + 2 * (self.A_21 - self.A_12) * x_A),
                            x_A ** 2 * (self.A_21 + 2 * (self.A_12 - self.A_21) * x_B)])

    def thermodynamic_factor(self, x):
        x_A, x_B = x
        return 1.0 - 2 * x_A * x_B * (self.A_12 * (2 * x_B - x_A) + self.A_21 * (2 * x_A - x_B))


class SoaveRedlichKwong:
    def __init__(self, R, T, P, v_ref, T_c, P_c, omega, l=0.0, l_b=0.0):
        self.RT, self.P, self.v_ref = float(R) * float(T), float(P), float(v_ref)
        T_c, P_c, omega = numpy.asarray([T_c, P_c, omega], dtype=float)
        alpha_SRK = (1.0 + (0.480 + 1.574 * omega - 0.176 * omega ** 2)
                     * (1.0 - numpy.sqrt(T / T_c))) ** 2
        a = 0.42748 * alpha_SRK * (R * T_c) ** 2 / P_c
        b = 0.08664 * R * T_c / P_c
        a_AB = (1.0 - l) * numpy.sqrt(a[0] * a[1])
        b_AB = (1.0 - l_b) * 0.5 * (b[0] + b[1])
        self.a_ij = numpy.array([[a[0], a_AB], [a_AB, a[1]]])
        self.b_ij = numpy.array([[b[0], b_AB], [b_AB, b[1]]])
        pure = numpy.eye(2)
        self.v_pure = self.liquid_volume(pure)[0]
        self.G_R_pure = self.residual_gibbs(*self.mixing(pure)[:2], self.v_pure)

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

    def root(self, a, b):
        coefficients = numpy.stack([self.P * numpy.ones_like(b), -self.RT * numpy.ones_like(b),
                                    a - self.RT * b - self.P * b ** 2, -a * b],
                                   axis=-1).reshape(-1, 4)
        v = numpy.array([numpy.roots(c) for c in coefficients], dtype=complex)
        v = numpy.where(v.imag == 0, v.real, numpy.inf)
        return v.min(axis=1).reshape(numpy.shape(b))

    @composition_cache
    def liquid_volume(self, x):
        a, b, a_x, b_x, a_xx, b_xx = self.mixing(x)
        P, RT = self.P, self.RT
        v = self.root(a, b)
        C_v = 3.0 * P * v ** 2 - 2.0 * RT * v + a - RT * b - P * b ** 2
        v_x = (((RT + 2.0 * P * b) * v + a) * b_x - (v - b) * a_x) / C_v
        v_xx = -((6.0 * P * v - 2.0 * RT) * v_x ** 2
                 + 2.0 * (a_x - (RT + 2.0 * P * b) * b_x) * v_x
                 - 2.0 * b_x * (a_x + P * v * b_x)
                 + (v - b) * a_xx
                 - ((RT + 2.0 * P * b) * v + a) * b_xx) / C_v
        return v, v_x, v_xx

    def residual_gibbs(self, a, b, v):
        P_RT = self.P / self.RT
        return (P_RT * v - 1.0 - numpy.log(P_RT * (v - b))
                - a / (b * self.RT) * numpy.log(1.0 + b / v))

    def molar_volume(self, x):
        v, v_x, v_xx = self.liquid_volume(x)
        return v / self.v_ref, v_x / self.v_ref, v_xx / self.v_ref

    def excess_volume(self, x):
        v, v_x, v_xx = self.liquid_volume(x)
        return ((v - x[0] * self.v_pure[0] - x[1] * self.v_pure[1]) / self.v_ref,
                (v_x - self.v_pure[0] + self.v_pure[1]) / self.v_ref,
                v_xx / self.v_ref)

    def excess_gibbs(self, x):
        a, b, a_x, b_x, a_xx, b_xx = self.mixing(x)
        v, v_x, _ = self.liquid_volume(x)
        P_RT = self.P / self.RT
        ratio, ln_1_b_v = a / (b * self.RT), numpy.log(1.0 + b / v)
        ln_a_x, ln_b_x = a_x / a, b_x / b
        G = self.residual_gibbs(a, b, v) - x[0] * self.G_R_pure[0] - x[1] * self.G_R_pure[1]
        G_x = ((P_RT * v - 1.0) * ln_b_x - ratio * ln_1_b_v * (ln_a_x - ln_b_x)
               - (self.G_R_pure[0] - self.G_R_pure[1]))
        G_xx = (P_RT * v_x * ln_b_x + (P_RT * v - 1.0) * (b_xx / b - ln_b_x ** 2)
                - ratio * ln_1_b_v * (a_xx / a - b_xx / b - 2.0 * ln_b_x * (ln_a_x - ln_b_x))
                - a / (self.RT * (v + b)) * (ln_b_x - v_x / v) * (ln_a_x - ln_b_x))
        return G, G_x, G_xx

    def ln_gamma(self, x):
        G, G_x, _ = self.excess_gibbs(x)
        return numpy.array([G + x[1] * G_x, G - x[0] * G_x])

    def thermodynamic_factor(self, x):
        _, _, G_xx = self.excess_gibbs(x)
        return 1.0 + x[0] * x[1] * G_xx


class Mixture:
    def __init__(self, phi_g_sat, alpha, theta, v_i, D0, excess_volume,
                 activity=None):
        self.phi_g_sat = numpy.asarray(phi_g_sat, dtype=float)
        self.alpha = numpy.asarray(alpha, dtype=float)
        self.theta = numpy.asarray(theta, dtype=float)
        self.v_i = numpy.asarray(v_i, dtype=float)
        self.D0 = numpy.asarray(D0, dtype=float)
        if self.phi_g_sat.size != 2:
            raise ValueError(f"Give two solvents, not {self.phi_g_sat.size}")
        self.activity = Ideal() if activity is None else activity
        self.excess_volume = Ideal() if excess_volume is None else excess_volume

    def molar_volume(self, x):
        if not isinstance(self.excess_volume, Ideal):
            return self.excess_volume.molar_volume(x)
        v = x[0] * self.v_i[0] + x[1] * self.v_i[1]
        v_x = self.v_i[0] - self.v_i[1]
        return v, v_x, 0.0

    def partial_volumes(self, x):
        v, v_x, _ = self.molar_volume(x)
        return numpy.array([v + x[1] * v_x, v - x[0] * v_x])

    def ideal_mole_fraction(self, phi_A):
        C_A, C_B = phi_A / self.v_i[0], (1.0 - phi_A) / self.v_i[1]
        return C_A / (C_A + C_B)

    def mole_fractions_phi(self, phi_A):
        def phi_A_error(x_A):
            x = numpy.array([x_A, 1.0 - x_A])
            v, _, _ = self.molar_volume(x)
            V_bar_A, _ = self.partial_volumes(x)
            return x_A * V_bar_A / v - phi_A

        x_A = self.ideal_mole_fraction(phi_A)
        if not isinstance(self.excess_volume, Ideal):
            x_A = scipy.optimize.newton(phi_A_error, x_A, tol=1e-15)
        return numpy.array([x_A, 1.0 - x_A])

    def s_x(self, x):
        v, _, v_xx = self.molar_volume(x)
        V_bar_A, V_bar_B = self.partial_volumes(x)
        curvature = x[0] * x[1] * v * v_xx
        return curvature / (V_bar_A * V_bar_B + curvature)

    def vapour_fraction_x(self, x):
        return self.phi_g_sat * numpy.exp(self.activity.ln_gamma(x)) * x

    def diffusivity_x(self, x):
        D_MS = self.D0[1] ** x[0] * self.D0[0] ** x[1]
        return D_MS * self.activity.thermodynamic_factor(x)
