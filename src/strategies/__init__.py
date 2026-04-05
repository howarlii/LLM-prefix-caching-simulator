"""Eviction strategies for the KV cache simulator."""

from src.strategies.base import EvictionStrategy
from src.strategies.crf_decoupling import CRFDecouplingStrategy
from src.strategies.fifo import FIFOStrategy
from src.strategies.lfu import LFUStrategy
from src.strategies.lru import LRUStrategy
from src.strategies.marconi import MarconiStrategy
from src.strategies.marconi2 import Marconi2Strategy

__all__ = [
    "EvictionStrategy",
    "CRFDecouplingStrategy",
    "LRUStrategy",
    "LFUStrategy",
    "FIFOStrategy",
    "MarconiStrategy",
    "Marconi2Strategy",
]
