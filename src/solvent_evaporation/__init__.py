"""Two-component solvent evaporation into still air, by the method of lines."""

from .mesh import Grid
from .mixture import Ideal, Margules, Mixture
from .model import EvaporationModel
from .compute import Result, compute

__all__ = ["Grid", "Ideal", "Margules", "Mixture", "EvaporationModel",
           "Result", "compute"]

__version__ = "0.1.0"
