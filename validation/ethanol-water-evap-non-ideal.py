import os

import numpy

from solvent_evaporation import (EvaporationModel, Grid, Margules, Mixture,
                                 SoaveRedlichKwong, compute_adaptive)

R = 8.314462618
T = 298.15
names = ["ethanol", "water"]
M = numpy.array([46.068e-3, 18.015e-3])
rho_pure = numpy.array([784.95, 997.05])
p_sat = numpy.array([7869.0, 3169.9])
Dg = numpy.array([1.29e-5, 2.54e-5])
volume_fractions = numpy.array([0.5, 0.5])

D0 = numpy.array([1.24e-9, 1.13e-9])

T_c = numpy.array([515, 647])
P_c = numpy.array([62.5e5, 220.5e5])
omega = numpy.array([0.645, 0.345])

l_AB, l_b_AB = -0.07865, 0.03972

A12, A21 = 1.5168, 0.9785

D_ref = 1.24e-9

delta0 = 1e-3

P_amb = 101325.0
gas_column = 1e-3

v = M / rho_pure
srk = SoaveRedlichKwong(R, T, P_amb, v[0], T_c, P_c, omega, l=l_AB, l_b=l_b_AB)
mixture = Mixture(phi_g_sat=p_sat / P_amb,
                  alpha=D_ref / Dg,
                  theta=v * P_amb / (R * T),
                  V_bar=v / v[0],
                  D0=D0 / D_ref,
                  activity=Margules(A12, A21),
                  excess_volume=srk)

phi0 = volume_fractions
phi0 = phi0 / phi0.sum()

b = delta0 + gas_column
b_hat = b / delta0

t_hat_end = 2.0
delta_hat_end = 1e-3
rtol, atol = 1e-6, 1e-9

dt_store = 0.05
dx = 0.005

liquid_grid = Grid(step=dx)
gas_grid = Grid(step=dx, start=1.0)

model = EvaporationModel(mixture, phi0, b_hat=b_hat,
                         liquid_grid=liquid_grid, gas_grid=gas_grid)

result = compute_adaptive(model, t_hat_end, dt_store=dt_store, rtol=rtol, atol=atol,
                          progress=10, delta_end=delta_hat_end)

output_dir = (f"{phi0[0]:.2f}_rtol_{rtol:.0e}_dx_{liquid_grid.step:.2e}-"
              f"t_hat_{t_hat_end:.1f}_b{b_hat:.1f}_ethanol-water-non-ideal")
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
                  header="eta_gas, then vapour mole fraction at each output time")
