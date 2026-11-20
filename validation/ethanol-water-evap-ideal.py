import os

import numpy

from solvent_evaporation import EvaporationModel, Grid, Mixture, compute_adaptive

R = 8.314462618
T = 298.15
names = ["ethanol", "water"]
M = numpy.array([46.04e-3, 18.015e-3])
rho_pure = numpy.array([785.0, 1000.0])
p_sat = numpy.array([7890.0, 3167.0])
Dg = numpy.array([1.1e-5, 2.4e-5])
mass_fractions = numpy.array([0.5, 0.5])

D0 = numpy.array([1.24e-9, 1.13e-9])

D_ref = 1.24e-9

delta0 = 1e-3

P_amb = 101325.0
gas_column = 1.0e-3

v = M / rho_pure
mixture = Mixture(phi_g_sat=p_sat / P_amb,
                  alpha=D_ref / Dg,
                  theta=v * P_amb / (R * T),
                  v_i=v / v[0],
                  D0=D0 / D_ref,
                  excess_volume=None)

phi0 = mass_fractions / rho_pure
phi0 = phi0 / phi0.sum()

b = delta0 + gas_column
b_hat = b / delta0

t_hat_end = 1.0
rtol, atol = 1e-6, 1e-9

dt_store = 0.005
dx = 0.05

liquid_grid = Grid(step=dx)
gas_grid = Grid(step=dx, start=1.0)

model = EvaporationModel(mixture, phi0, b_hat=b_hat,
                         liquid_grid=liquid_grid, gas_grid=gas_grid)

result = compute_adaptive(model, t_hat_end, dt_store=dt_store, rtol=rtol, atol=atol,
                          progress=10)

output_dir = (f"{phi0[0]:.2f}_rtol_{rtol:.0e}_dx_{liquid_grid.step:.2e}-"
              f"t_hat_{t_hat_end:.1f}_b{b_hat:g}_ethanol-water-ideal")
os.makedirs(output_dir, exist_ok=True)

numpy.savetxt(f"{output_dir}/delta_time_mol.dat",
              numpy.column_stack((result.t, result.delta)),
              header="t_hat delta_hat")
numpy.savetxt(f"{output_dir}/flux_mol.dat",
              numpy.column_stack((result.t, result.flux.T)),
              header="t_hat " + " ".join(f"n_{name}" for name in names))
for i, name in enumerate(names):
    numpy.savetxt(f"{output_dir}/phi_{name}_mol.dat",
                  numpy.column_stack((result.eta_liquid, result.phi[i])),
                  header="eta_liquid, then volume fraction at each output time")
    numpy.savetxt(f"{output_dir}/phi_g_{name}_mol.dat",
                  numpy.column_stack((result.eta_gas, result.phi_g[i])),
                  header="eta_gas, then gas volume fraction at each output time")
