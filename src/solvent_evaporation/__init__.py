from .mesh import Grid
from .mixture import Ideal, Margules, Mixture, SoaveRedlichKwong
from .model import EvaporationModel
from .compute import Result, compute_fixed, compute_adaptive

__all__ = ["Grid", "Ideal", "Margules", "Mixture", "SoaveRedlichKwong",
           "EvaporationModel", "Result", "compute_fixed", "compute_adaptive"]

__version__ = "0.1.0"
