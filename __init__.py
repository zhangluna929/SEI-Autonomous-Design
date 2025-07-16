"""
SEI Autonomous Design Platform

Multi-modal solid-state interface autonomous design platform for 
solid electrolyte interface (SEI) design.
"""

__version__ = "1.0.0"
__author__ = "lunazhang"

from . import pretrain
from . import finetune  
from . import generator
from . import active_learning
from . import multiscale
from . import rl_generation
from . import data_simulate

__all__ = [
    "pretrain",
    "finetune", 
    "generator",
    "active_learning",
    "multiscale",
    "rl_generation",
    "data_simulate"
] 
