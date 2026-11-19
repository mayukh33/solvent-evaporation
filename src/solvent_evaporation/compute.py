import time
from dataclasses import dataclass

import numpy
import scipy


def show_progress(t, t_end, delta, started):
    done = t / t_end
    elapsed = time.perf_counter() - started
    filled = int(30 * done)
    print(f"\r[{'#' * filled}{'.' * (30 - filled)}] {done:5.1%}  "
          f"t_hat {t:.4g}  delta_hat {delta:.5f}  "
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
    dy = eps * numpy.abs(y)
    dy[dy == 0.0] = eps
    return scipy.optimize.approx_fprime(
        y, lambda perturbed: model.rhs(t, perturbed), dy)


def backward_euler_step(model, t, y, dt):
    f = model.rhs(t, y)
    dfdy = jacobian(model, t, y)
    return y + dt * numpy.linalg.solve(numpy.eye(y.size) - dt * dfdy, f)


def compute_fixed(model, t_end, dt, store_period=1, progress=0, delta_end=None):
    n_steps = numpy.max((1, numpy.round(t_end / dt))).astype(numpy.int64)
    delta_end = 0 if delta_end is None else delta_end
    y = model.initial_state()
    stored = {0: y}
    started = time.perf_counter()
    try:
        for k in range(1, n_steps + 1):
            y = backward_euler_step(model, (k - 1) * dt, y, dt)
            thin = y[model.delta_index] <= delta_end
            if k % store_period == 0 or k == n_steps or thin:
                stored[k] = y
            if progress and k % progress == 0:
                show_progress(k * dt, n_steps * dt, y[model.delta_index], started)
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

    t = dt * numpy.array(list(stored))
    return post_process(model, t, numpy.column_stack(list(stored.values())))


def compute_adaptive(model, t_end, dt_store, rtol, atol, progress=0, delta_end=None):
    solver = scipy.integrate.BDF(model.rhs, 0.0, model.initial_state(), t_end,
                                 rtol=rtol, atol=atol,
                                 jac=lambda t, y: jacobian(model, t, y))
    t_store = dt_store * numpy.arange(1, numpy.round(t_end / dt_store) + 1)
    delta_end = 0 if delta_end is None else delta_end

    stored = {0.0: solver.y}
    started = time.perf_counter()
    k, message = 0, None
    try:
        while solver.status == "running" and solver.y[model.delta_index] > delta_end:
            message = solver.step()
            if message:
                break
            k += 1
            due = t_store[(t_store > solver.t_old) & (t_store <= solver.t)]
            stored.update(zip(due, solver.dense_output()(due).T))
            if progress and k % progress == 0:
                show_progress(solver.t, t_end, solver.y[model.delta_index], started)
    except KeyboardInterrupt:
        message = "interrupted"
    stored[solver.t] = solver.y
    if progress:
        print()
    if solver.status != "finished":
        print(f"stopped at step {k} (t_hat = {solver.t:.6g}, delta_hat = "
              f"{solver.y[model.delta_index]:.6g}): {message or 'delta_end reached'}; "
              f"returning {len(stored)} stored times")

    t = numpy.array(list(stored))
    return post_process(model, t, numpy.column_stack(list(stored.values())))


def post_process(model, t, states):
    num_components = 2
    n_times = t.size
    delta = numpy.empty(n_times)
    phi = numpy.empty((num_components, model.liquid_grid.n + 1, n_times))
    phi_g = numpy.empty((num_components, model.gas_grid.n + 1, n_times))
    phi_interface = numpy.empty((num_components, n_times))
    flux = numpy.empty((num_components, n_times))

    for k in range(n_times):
        phi[..., k], phi_g[..., k], delta[k], _ = model.fields(states[:, k])
        phi_interface[:, k], flux[:, k] = model.surface(states[:, k])

    return Result(model=model, t=t, delta=delta, phi=phi, phi_g=phi_g,
                  phi_interface=phi_interface, flux=flux)