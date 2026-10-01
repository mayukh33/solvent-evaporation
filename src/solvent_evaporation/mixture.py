import numpy
from . import diffusion


class Ideal:
    """Ideal solution.  An activity model is any object with these two methods."""

    def ln_gamma(self, x):
        """ln gamma_i, (N,), at liquid mole fractions x (N,)."""
        return numpy.zeros_like(x)

    def thermodynamic_factor(self, x):
        """[Gamma], (N-1, N-1), at mole fractions x (N,), with x_N eliminated."""
        # Gamma_ij = delta_ij + x_i d(ln gamma_i)/dx_j
        return numpy.eye(x.size - 1)


class Margules:
    """Binary two-parameter Margules: A12 = ln gamma_1^inf, A21 = ln gamma_2^inf."""

    def __init__(self, A12, A21):
        self.A12, self.A21 = float(A12), float(A21)

    def ln_gamma(self, x):
        """ln gamma_i, (2,), at liquid mole fractions x (2,)."""
        # ln gamma_1 = x_2^2 [A12 + 2 (A21 - A12) x_1]
        # ln gamma_2 = x_1^2 [A21 + 2 (A12 - A21) x_2]
        x1, x2 = x
        return numpy.array([x2 ** 2 * (self.A12 + 2 * (self.A21 - self.A12) * x1),
                            x1 ** 2 * (self.A21 + 2 * (self.A12 - self.A21) * x2)])

    def thermodynamic_factor(self, x):
        """[Gamma], (1, 1), at mole fractions x (2,)."""
        # Gamma = 1 + x_1 d(ln gamma_1)/dx_1
        #       = 1 - 2 x_1 x_2 [A12 (2 x_2 - x_1) + A21 (2 x_1 - x_2)]
        x1, x2 = x
        return numpy.array([[1.0 - 2 * x1 * x2 * (self.A12 * (2 * x2 - x1)
                                                   + self.A21 * (2 * x1 - x2))]])


class Mixture:
    """Two solvents as dimensionless groups; either order is accepted.

    The components are known by name.  The first is independent; the second
    is the balance, whose volume fraction is whatever the first leaves.
    """

    def __init__(self, psi_sat, alpha, theta, V_bar, D0, activity=None,
                 names=None):
        self.psi_sat = numpy.asarray(psi_sat, dtype=float)
        self.alpha = numpy.asarray(alpha, dtype=float)
        self.theta = numpy.asarray(theta, dtype=float)
        self.V_bar = numpy.asarray(V_bar, dtype=float)
        self.D0 = numpy.asarray(D0, dtype=float)
        self.n = self.psi_sat.size
        # the interface solve brackets a single unknown: two components only
        if self.n != 2:
            raise ValueError(f"Give two "
                             f"solvents, not {self.n}")
        self.activity = Ideal() if activity is None else activity

        if names is None:
            names = [f"component {i + 1}" for i in range(self.n)]
        self.names = list(names)
        if len(self.names) != self.n or len(set(self.names)) != self.n:
            raise ValueError(f"give {self.n} distinct component names, "
                             f"not {self.names}")
        self.components = self.names[:-1]
        self.balance = self.names[-1]

    def volume_fractions(self, components, tol=1e-12):
        """phi, (N, ...) in name order, from the independent phi_i by name.

        The balance component makes up the rest, and the result is checked to
        sum to one.
        """
        if set(components) != set(self.components):
            raise ValueError(f"give the independent components "
                             f"{self.components}, not {list(components)}")
        phi = {}
        for name in self.components:
            phi[name] = numpy.asarray(components[name], dtype=float)

        phi[self.balance] = 1.0
        for name in self.components:
            phi[self.balance] = phi[self.balance] - phi[name]

        total = 0.0
        for name in self.names:
            total = total + phi[name]
        error = numpy.max(numpy.abs(total - 1.0))
        # not (error <= tol), so that a NaN fails as well
        if not error <= tol:
            raise ValueError(f"volume fractions of {' + '.join(self.names)} "
                             f"do not sum to one: off by {error:.3g}")

        result = numpy.empty((self.n,) + numpy.shape(phi[self.balance]))
        for i, name in enumerate(self.names):
            result[i] = phi[name]
        return result

    def mole_fractions(self, phi):
        """Mole fractions at phi (N,), normalised over components; not clipped."""
        C_i = phi / self.V_bar
        return C_i / C_i.sum()

    def vapour_fraction(self, phi):
        """Modified Raoult's law: psi_i = psi_sat,i gamma_i x_i."""
        # psi_i^s = psi_sat,i gamma_i x_i^(l),  psi_sat,i = p_sat,i V_bar_i / (R T)
        x = self.mole_fractions(phi)
        return self.psi_sat * numpy.exp(self.activity.ln_gamma(x)) * x

    def fick_matrix(self, phi):
        """Volume-frame Fick matrix over D_ref, (N-1, N-1), at phi (N,)."""
        Gamma = self.activity.thermodynamic_factor(self.mole_fractions(phi))
        return diffusion.fick_matrix(phi, self.V_bar, self.D0, Gamma)
