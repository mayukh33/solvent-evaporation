from .mesh import Grid
from .mixture import Ideal, Margules, Mixture, SoaveRedlichKwong
from .model import EvaporationModel
from .compute import Result, compute

__all__ = ["Grid", "Ideal", "Margules", "Mixture", "SoaveRedlichKwong",
           "EvaporationModel", "Result", "compute"]

__version__ = "0.1.0"
