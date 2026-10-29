import numpy

try:
    from numba import njit
except ImportError:
    def njit(function=None, **options):
        return function if function is not None else njit


@njit(cache=True)
def maxwell_stefan_diffusivities(x, D0):
    """Vignes diffusivities D_ij, (N, N), at mole fractions x.  Diagonal unused."""
    # D_ij = D0_ji^x_i D0_ij^x_j prod_{k != i,j} (D0_ik D0_jk)^(x_k/2)
    num_components = x.size
    D_ij = numpy.ones((num_components, num_components))
    for i in range(num_components):
        for j in range(num_components):
            if i == j:
                continue
            D_ij[i, j] = D0[j, i] ** x[i] * D0[i, j] ** x[j]
            for k in range(num_components):
                if k != i and k != j:
                    D_ij[i, j] *= (D0[i, k] * D0[j, k]) ** (0.5 * x[k])
    return D_ij


@njit(cache=True)
def mole_fraction_jacobian(phi, V_bar):
    """X_jl = dx_j/dphi_l, (N-1, N-1), at volume fractions phi, with phi_N eliminated."""
    # X_jl = delta_jl/(V_bar_j C) - (C_j/C^2)(1/V_bar_l - 1/V_bar_N)
    n = phi.size - 1
    C_i = phi / V_bar
    C = C_i.sum()
    return (numpy.diag(1.0 / V_bar[:n]) / C
            - numpy.outer(C_i[:n] / C ** 2, 1.0 / V_bar[:n] - 1.0 / V_bar[-1]))


@njit(cache=True)
def fick_matrix(phi, V_bar, D0, Gamma):
    """Volume-frame Fick matrix over D_ref, (N-1, N-1), at volume fractions phi."""
    # j = -[D] grad phi ,  [D] = diag(V_bar) T C B^-1 Gamma X
    num_components = phi.size
    n = num_components - 1
    C_i = phi / V_bar
    C = C_i.sum()
    x = C_i / C
    inv_D_ij = 1.0 / maxwell_stefan_diffusivities(x, D0)

    # B_ii = x_i/D_iN + sum_{k != i} x_k/D_ik ;  B_ij = -x_i (1/D_ij - 1/D_iN)
    B = numpy.empty((n, n))
    for i in range(n):
        others = 0.0
        for k in range(num_components):
            if k != i:
                others += x[k] * inv_D_ij[i, k]
        B[i, i] = x[i] * inv_D_ij[i, num_components - 1] + others
        for j in range(n):
            if j != i:
                B[i, j] = -x[i] * (inv_D_ij[i, j]
                                   - inv_D_ij[i, num_components - 1])

    # T_ik = delta_ik - C_i (V_bar_k - V_bar_N): molar frame -> volume frame
    T = numpy.eye(n) - numpy.outer(C_i[:n], V_bar[:n] - V_bar[-1])
    X = mole_fraction_jacobian(phi, V_bar)
    fick = numpy.diag(V_bar[:n]) @ T
    fick = fick @ (C * numpy.linalg.inv(B))
    fick = fick @ Gamma
    return fick @ X
