"""Fixed-step backward Euler."""

import time
from dataclasses import dataclass

import numpy


def show_progress(k, n_steps, t_hat, delta_hat, started):
    """One line, rewritten in place: bar, position, and time left."""
    done = k / n_steps
    elapsed = time.perf_counter() - started
    filled = int(30 * done)
    print(f"\r[{'#' * filled}{'.' * (30 - filled)}] {done:5.1%}  "
          f"t_hat {t_hat:.4g}  delta_hat {delta_hat:.5f}  "
          f"{elapsed:5.0f}s elapsed",
          end="", flush=True)


@dataclass
class Result:
    """Simulation output, all scaled.  Profiles are [component, cell, time]."""

    model: object
    t: numpy.ndarray               # t_hat
    delta: numpy.ndarray           # delta_hat
    phi: numpy.ndarray             # liquid volume fractions
    psi: numpy.ndarray             # vapour, liquid-equivalent
    phi_interface: numpy.ndarray
    flux: numpy.ndarray            # evaporative flux n_i

    @property
    def eta_liquid(self):
        return self.model.liquid_grid.centers

    @property
    def eta_gas(self):
        return self.model.gas_grid.centers


def jacobian(model, t, y, f0, eps=1e-7):
    """df/dy by forward differences about y, where f0 = f(t, y) already is.
    """
    # J[:, i] = (f(y + h e_i) - f(y)) / h,  h = eps |y_i|
    J = numpy.empty((y.size, y.size))
    for i in range(y.size):
        h = eps * abs(y[i])
        if h == 0.0:
            h = eps
        perturbed = y.copy()
        perturbed[i] = y[i] + h
        h = perturbed[i] - y[i]
        J[:, i] = (model.rhs(t, perturbed) - f0) / h
    return J


def backward_euler_step(model, t, y, h):
    """The state at t + h, by the semi-implicit Euler step of Numerical Recipes.

    y_next = y + h [1 - h df/dy]^-1 . f(y)
    """
    f = model.rhs(t, y)
    dfdy = jacobian(model, t, y, f)
 
    inverse = numpy.linalg.inv(numpy.eye(y.size) - h * dfdy)
    return y + h * numpy.matmul(inverse, f)


def compute(model, t_end, dt, store_period=1, progress=0):
    """March from t_hat = 0 to t_end in steps of exactly dt."""
    n_steps = numpy.max((1, numpy.round(t_end / dt))).astype(numpy.int64)

    # Worked out up front so states is allocated once at its final size.
    keep = numpy.unique(numpy.append(numpy.arange(0, n_steps + 1, store_period),
                                     n_steps))
    states = numpy.empty((model.size, keep.size))

    y = model.initial_state()
    states[:, 0] = y
    kept = 1

    started = time.perf_counter()
    try:
        for k in range(1, n_steps + 1):
            # (k - 1) * dt, not a running sum, which drifts over many steps.
            y = backward_euler_step(model, (k - 1) * dt, y, dt)
            if kept < keep.size and k == keep[kept]:
                states[:, kept] = y
                kept += 1
            if progress and k % progress == 0:
                show_progress(k, n_steps, k * dt, y[model.delta_index], started)
    except KeyboardInterrupt:
        # Keyboard interruption does not result in loss of data.
        print(f"\ninterrupted at step {k} of {n_steps} (t_hat = {k * dt:.6g}); "
              f"returning {kept} stored times, to t_hat = "
              f"{keep[kept - 1] * dt:.6g}")
    else:
        if progress:
            print()

    return post_process(model, keep[:kept] * dt, states[:, :kept])


def post_process(model, t, states):
    """Split each stored state into profiles and interface values."""
    m = model.mixture
    nt = t.size
    delta = numpy.empty(nt)
    phi = numpy.empty((m.n, model.liquid_grid.n, nt))
    psi = numpy.empty((m.n, model.gas_grid.n, nt))
    phi_interface = numpy.empty((m.n, nt))
    flux = numpy.empty((m.n, nt))

    for c in range(nt):
        fields = model.fields(states[:, c])
        phi[..., c], psi[..., c], delta[c], L_hat = fields
        phi_interface[:, c], flux[:, c], _, _ = model.interface(
            phi[:, -1, c], psi[:, 0, c], delta[c], L_hat)

    return Result(model=model, t=t, delta=delta, phi=phi, psi=psi,
                  phi_interface=phi_interface, flux=flux)
