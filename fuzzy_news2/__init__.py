"""
Fuzzy logic implementation of the National Early Warning Score 2 (NEWS-2).
"""

from .news2 import FuzzyNEWS2, NEWS2Result
from .fuzzy_logic import FuzzyLogic

__version__ = "0.1.0"
__all__ = ["FuzzyNEWS2", "NEWS2Result", "FuzzyLogic"]

