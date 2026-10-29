"""Fixtures shared by the tests, after flyft's python/tests/conftest.py: each
sets up one part of the model, the mixture, the meshes and the model itself,
with its numbers inside it, for a test to take by name.  The binary is made
up, with round numbers and every quantity over its scale."""

import numpy
import pytest

from solvent_evaporation import EvaporationModel, Grid, Ideal, Mixture


@pytest.fixture
def ideal():
    V_bar = numpy.array([1.0, 2.0])             # molar volumes, B's twice A's
    D0 = numpy.array([[0.0, 1.0], [3.0, 0.0]])  # A dilute in B, B dilute in A
    D_g = numpy.array([2.0, 4.0])               # each vapour in air
    phi_g_sat = numpy.array([0.1, 0.005])       # the interface's only
    theta = numpy.array([1e-2, 2e-2])           # the interface's only
    return Mixture(phi_g_sat=phi_g_sat, alpha=1.0 / D_g, theta=theta,
                   V_bar=V_bar, D0=D0, activity=Ideal(), names=["A", "B"])


@pytest.fixture
def liquid_grid():
    return Grid(n=500)


@pytest.fixture
def gas_grid():
    return Grid(n=500, start=1.0)


@pytest.fixture(params=[1.0, 0.4])
def delta_hat(request):
    """The film as cast and thinned; the gas fills the rest to b_hat."""
    return request.param


@pytest.fixture
def model(ideal, liquid_grid, gas_grid):
    return EvaporationModel(ideal, [0.5, 0.5], b_hat=2.0,
                            liquid_grid=liquid_grid, gas_grid=gas_grid)
