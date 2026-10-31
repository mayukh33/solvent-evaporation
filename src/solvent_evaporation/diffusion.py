import numpy

try:
    from numba import njit
except ImportError:
    def njit(function=None, **options):
        return function if function is not None else njit


@njit(cache=True)
def maxwell_stefan_diffusivities(x, D0):
    N = x.size
    D = numpy.ones((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            D[i, j] = D0[j, i] ** x[i] * D0[i, j] ** x[j]
            for k in range(N):
                if k != i and k != j:
                    D[i, j] *= (D0[i, k] * D0[j, k]) ** (0.5 * x[k])
    return D
