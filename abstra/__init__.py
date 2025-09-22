"""ABSTRA: Abstract Section-Targeted Reasoning Assessment Framework"""

__version__ = "0.1.0"

from .pipeline import ABSTRAPipeline
from .config import Config

__all__ = ["ABSTRAPipeline", "Config"]