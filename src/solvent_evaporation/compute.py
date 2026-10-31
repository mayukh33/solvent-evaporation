import time
from dataclasses import dataclass

import numpy
import scipy


def show_progress(k, n_steps, t_hat, delta_hat, started):
    done = k / n_steps
    elapsed = time.perf_counter() - started
    filled = int(30 * done)
    print(f"\r[{'#' * filled}{'.' * (30 - filled)}] {done:5.1%}  "
          f"t_hat {t_hat:.4g}  delta_hat {delta_hat:.5f}  "
          f"{elapsed:5.0f}s elapsed",
          end="", flush=True)


@dataclass
class Result:
    model: object
    t: numpy.ndarray
    delta: numpy.ndarray
    phi: numpy.ndarray
    phi_g: numpy.ndarray
    phi_interface: numpy.ndarray
    flux: numpy.ndarray

    @property
    def eta_liquid(self):
        return self.model.liquid_grid.faces

    @property
    def eta_gas(self):
        return self.model.gas_grid.faces


def jacobian(model, t, y, eps=1e-7):
    h = eps * numpy.abs(y)
    h[h == 0.0] = eps
    return scipy.optimize.approx_fprime(
        y, lambda perturbed: model.rhs(t, perturbed), h)


def backward_euler_step(model, t, y, h):
    f = model.rhs(t, y)
    dfdy = jacobian(model, t, y)
    return y + h * numpy.linalg.solve(numpy.eye(y.size) - h * dfdy, f)


def compute(model, t_end, dt, store_period=1, progress=0, delta_end=None):
    n_steps = numpy.max((1, numpy.round(t_end / dt))).astype(numpy.int64)

    y = model.initial_state()
    stored = {0: y}
    started = time.perf_counter()
    try:
        for k in range(1, n_steps + 1):
            y = backward_euler_step(model, (k - 1) * dt, y, dt)
            thin = delta_end is not None and y[model.delta_index] <= delta_end
            if k % store_period == 0 or k == n_steps or thin:
                stored[k] = y
            if progress and k % progress == 0:
                show_progress(k, n_steps, k * dt, y[model.delta_index], started)
            if thin:
                break
    except KeyboardInterrupt:
        print(f"\ninterrupted at step {k} of {n_steps} (t_hat = {k * dt:.6g}); "
              f"returning {len(stored)} stored times, to t_hat = "
              f"{max(stored) * dt:.6g}")
    else:
        if progress:
            print()
        if thin:
            print(f"delta_hat = {y[model.delta_index]:.6g} <= {delta_end:g} at step "
                  f"{k} of {n_steps} (t_hat = {k * dt:.6g}); stopped there, "
                  f"returning {len(stored)} stored times")

    steps = numpy.array(list(stored))
    return post_process(model, steps * dt, numpy.column_stack(list(stored.values())))


def post_process(model, t, states):
    m = model.mixture
    nt = t.size
    delta = numpy.empty(nt)
    phi = numpy.empty((m.n, model.liquid_grid.n + 1, nt))
    phi_g = numpy.empty((m.n, model.gas_grid.n + 1, nt))
    phi_interface = numpy.empty((m.n, nt))
    flux = numpy.empty((m.n, nt))

    for c in range(nt):
        phi[..., c], phi_g[..., c], delta[c], _ = model.fields(states[:, c])
        phi_interface[:, c], flux[:, c] = model.surface(states[:, c])

    return Result(model=model, t=t, delta=delta, phi=phi, phi_g=phi_g,
                  phi_interface=phi_interface, flux=flux)