"""Eviction strategies for the KV cache simulator."""

from src.strategies.base import EvictionStrategy
from src.strategies.branch import BranchStrategy
from src.strategies.crf_decoupling import CRFDecouplingStrategy
from src.strategies.fifo import FIFOStrategy
from src.strategies.lru import LRUStrategy
from src.strategies.marconi import MarconiStrategy
from src.strategies.marconi2 import Marconi2Strategy
from src.strategies.marconi3 import Marconi3Strategy
from src.strategies.oracle_greedy import OracleGreedyStrategy

__all__ = [
    "EvictionStrategy",
    "BranchStrategy",
    "CRFDecouplingStrategy",
    "LRUStrategy",
    "FIFOStrategy",
    "MarconiStrategy",
    "Marconi2Strategy",
    "Marconi3Strategy",
    "OracleGreedyStrategy",
]
